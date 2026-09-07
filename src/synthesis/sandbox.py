from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import shutil
import tempfile
import time
from typing import Mapping

from .schemas import RunResult


@dataclass(slots=True)
class SandboxConfig:
    timeout_sec: float = 3.0
    memory_mb: int = 512
    max_output_bytes: int = 65536
    max_input_bytes: int = 65536
    max_processes: int = 8
    python_executable: str = sys.executable
    backend: str = "local"


_NETWORK_GUARD = r'''# installed before candidate execution
import socket as _synthesis_socket
def _synthesis_network_denied(*args, **kwargs):
    raise PermissionError("network access is disabled")
_synthesis_socket.socket = _synthesis_network_denied
_synthesis_socket.create_connection = _synthesis_network_denied
'''


def _limits(config: SandboxConfig):
    def apply() -> None:
        os.setsid()
        memory = config.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        cpu = max(1, int(config.timeout_sec) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_NPROC, (config.max_processes, config.max_processes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def _sandbox_command(
    config: SandboxConfig, temporary: str, script: Path,
) -> list[str]:
    local = [
        config.python_executable, "-I", "-S", str(script),
    ]
    if config.backend == "local":
        return local
    if config.backend != "bwrap":
        raise ValueError(f"unknown sandbox backend: {config.backend}")
    executable = shutil.which("bwrap")
    if executable is None:
        raise RuntimeError("bwrap backend requested but bwrap is unavailable")
    command = [
        executable,
        "--die-with-parent",
        "--unshare-all",
        "--new-session",
        "--clearenv",
        "--setenv", "LANG", "C.UTF-8",
        "--setenv", "LC_ALL", "C.UTF-8",
        "--setenv", "PYTHONHASHSEED", "0",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
    ]
    for system_path in ("/usr", "/lib", "/lib64", "/usr/local"):
        if Path(system_path).exists():
            command.extend(["--ro-bind", system_path, system_path])
    command.extend([
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--ro-bind", temporary, "/work",
        "--chdir", "/work",
        config.python_executable,
        "-I", "-S", "/work/candidate.py",
    ])
    return command


class SandboxedExecutor:
    """Run Python with limits; formal runs use an OS-level bwrap boundary.

    The bwrap backend exposes only runtime libraries and the episode directory,
    unshares networking/PIDs/IPC, and never mounts grader or dataset files.
    """

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()

    def run(self, code: str, input_text: str, env: Mapping[str, str] | None = None) -> RunResult:
        cfg = self.config
        if not isinstance(code, str) or not isinstance(input_text, str):
            return RunResult(
                status="invalid_input",
                error={"type": "TypeError", "message": "code and input must be strings"},
                exit_code=None,
            )
        encoded_input = input_text.encode("utf-8")
        if len(encoded_input) > cfg.max_input_bytes:
            return RunResult(
                status="invalid_input", error={"type": "InputLimit", "message": "input exceeds byte limit"}, exit_code=None
            )
        try:
            compile(code, "<candidate>", "exec")
        except (SyntaxError, TypeError, ValueError) as exc:
            return RunResult(
                status="runtime_error", error={"type": "SyntaxError", "message": str(exc)}, exit_code=1
            )

        with tempfile.TemporaryDirectory(prefix="synthesis-episode-") as temporary:
            script = Path(temporary) / "candidate.py"
            script.write_text(_NETWORK_GUARD + "\n" + code, encoding="utf-8")
            safe_env = {
                "PATH": os.environ.get("PATH", ""),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            if env:
                safe_env.update(env)
            started = time.monotonic()
            try:
                process = subprocess.Popen(
                    _sandbox_command(cfg, temporary, script),
                    cwd=temporary,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=safe_env,
                    preexec_fn=_limits(cfg),
                )
                stdout, stderr = process.communicate(encoded_input, timeout=cfg.timeout_sec)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
                return self._result("timeout", stdout, stderr, None, {"type": "Timeout", "message": f"exceeded {cfg.timeout_sec}s"})
            except (OSError, ValueError) as exc:
                return RunResult(status="runtime_error", error={"type": type(exc).__name__, "message": str(exc)}, exit_code=None)

            del started  # retained as a natural audit hook without leaking into tool output
            if process.returncode != 0:
                return self._result(
                    "runtime_error", stdout, stderr, process.returncode,
                    {"type": "ProcessError", "message": "candidate exited unsuccessfully"},
                )
            return self._result("ok", stdout, stderr, process.returncode, None)

    def _result(self, status: str, stdout: bytes, stderr: bytes, exit_code: int | None, error: dict[str, str] | None) -> RunResult:
        limit = self.config.max_output_bytes
        truncated = len(stdout) > limit or len(stderr) > limit
        stdout = stdout[:limit]
        stderr = stderr[:limit]
        if truncated and status == "ok":
            status = "output_limit"
            error = {"type": "OutputLimit", "message": "stdout or stderr exceeded byte limit"}
        return RunResult(
            status=status,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            error=error,
            exit_code=exit_code,
            truncated=truncated,
        )

