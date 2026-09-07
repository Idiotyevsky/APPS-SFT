from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
import random
import re
import time
from typing import Any, Protocol
import urllib.error
import urllib.request

from .schemas import validate_tool_call


class TextGenerator(Protocol):
    name: str
    revision: str
    def generate(self, prompt: str, seed: int) -> str: ...


@dataclass(slots=True)
class GenerationArtifact:
    model: str
    revision: str
    seed: int
    prompt_hash: str
    raw_output: str

    @property
    def raw_hash(self) -> str:
        return "sha256:" + sha256(self.raw_output.encode()).hexdigest()


class TransformersTextGenerator:
    def __init__(self, name: str, revision: str, temperature: float = 0.7, top_p: float = 0.95, max_new_tokens: int = 4096, dtype: str = "bf16"):
        if not revision or revision.lower() == "latest":
            raise ValueError("a pinned model revision is required")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the model extra to use TransformersTextGenerator") from exc
        self.name, self.revision = name, revision
        self.temperature, self.top_p, self.max_new_tokens = temperature, top_p, max_new_tokens
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=torch_dtype, device_map="auto")
        self.calls = 0
        self.wall_seconds = 0.0

    def generate(self, prompt: str, seed: int) -> str:
        started = time.monotonic()
        self.calls += 1
        try:
            torch = self._torch
            torch.manual_seed(seed)
            encoded = self.tokenizer(
                prompt, return_tensors="pt",
            ).to(self.model.device)
            output = self.model.generate(
                **encoded,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-6),
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
            )
            generated = output[
                0, encoded["input_ids"].shape[1]:
            ]
            return self.tokenizer.decode(
                generated, skip_special_tokens=True,
            )
        finally:
            self.wall_seconds += time.monotonic() - started


class OpenRouterTextGenerator:
    """OpenRouter chat-completions generator with optional reasoning.

    The API key is read from the process environment and is kept in memory
    only. Provider reasoning_details is retained on the instance so a
    caller can include it unchanged in a follow-up message when needed.
    """

    def __init__(
        self,
        name: str = "z-ai/glm-5.3-flash",
        revision: str | None = None,
        *,
        api_key: str | None = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 4096,
        reasoning_enabled: bool = True,
        max_retries: int = 4,
    ) -> None:
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise ValueError(
                f"OpenRouter API key is missing; set {api_key_env}"
            )
        if not isinstance(name, str) or not name.strip():
            raise ValueError("OpenRouter model name is required")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.name = name
        self.revision = revision or f"openrouter:{name}"
        self._api_key = key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.reasoning_enabled = bool(reasoning_enabled)
        self.max_retries = max_retries
        self.calls = 0
        self.wall_seconds = 0.0
        self.last_reasoning_details: Any = None
        self.last_response: dict[str, Any] | None = None
        # In-memory audit only; callers may persist this without exposing the
        # API key. Each item contains request metadata and provider response.
        self.history: list[dict[str, Any]] = []

    def _request(self, messages: list[dict[str, Any]], seed: int) -> str:
        if not isinstance(messages, list) or any(
            not isinstance(message, dict) for message in messages
        ):
            raise TypeError("messages must be a list of objects")
        payload = {
            "model": self.name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_new_tokens,
            "seed": seed,
            "reasoning": {"enabled": self.reasoning_enabled},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        value: Any
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    value = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 or 500 <= exc.code < 600:
                    if attempt < self.max_retries:
                        retry_after = exc.headers.get("Retry-After", "")
                        try:
                            delay = min(60.0, max(1.0, float(retry_after)))
                        except (TypeError, ValueError):
                            delay = min(60.0, 2.0 ** attempt)
                        time.sleep(delay)
                        continue
                try:
                    detail = json.loads(body).get("error", {}).get(
                        "message", body,
                    )
                except (TypeError, json.JSONDecodeError):
                    detail = body[:500]
                raise RuntimeError(
                    f"OpenRouter HTTP {exc.code}: {str(detail)[:500]}"
                ) from exc
            except (OSError, TimeoutError) as exc:
                if attempt < self.max_retries:
                    time.sleep(min(60.0, 2.0 ** attempt))
                    continue
                raise RuntimeError(
                    f"OpenRouter request failed: {type(exc).__name__}: {exc}"
                ) from exc
        else:
            raise RuntimeError("OpenRouter request exhausted retries")
        if not isinstance(value, dict):
            raise RuntimeError("OpenRouter response is not an object")
        try:
            message = value["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "OpenRouter response has no assistant choice"
            ) from exc
        if not isinstance(message, dict):
            raise RuntimeError("OpenRouter assistant message is invalid")
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise RuntimeError("OpenRouter assistant content is not text")
        self.last_reasoning_details = message.get("reasoning_details")
        self.last_response = value
        self.history.append({
            "seed": seed,
            "request": {
                "model": self.name,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_new_tokens,
                "reasoning": {"enabled": self.reasoning_enabled},
            },
            "response": value,
        })
        return content

    def generate(self, prompt: str, seed: int) -> str:
        started = time.monotonic()
        self.calls += 1
        try:
            if not isinstance(prompt, str):
                raise TypeError("prompt must be a string")
            return self._request(
                [{"role": "user", "content": prompt}], seed,
            )
        finally:
            self.wall_seconds += time.monotonic() - started

    def generate_messages(
        self, messages: list[dict[str, Any]], seed: int,
    ) -> str:
        """Generate from messages, allowing reasoning_details continuation."""
        started = time.monotonic()
        self.calls += 1
        try:
            return self._request(messages, seed)
        finally:
            self.wall_seconds += time.monotonic() - started


def generate_artifact(generator: TextGenerator, prompt: str, seed: int) -> GenerationArtifact:
    raw = generator.generate(prompt, seed)
    digest = "sha256:" + sha256(prompt.encode()).hexdigest()
    return GenerationArtifact(generator.name, generator.revision, seed, digest, raw)


def _extract_json_object(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char == "{":
            try:
                value, end = decoder.raw_decode(stripped[index:])
                return json.dumps(value, ensure_ascii=False)
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object in model output")


def parse_tool_action(raw: str) -> dict[str, Any]:
    value = json.loads(_extract_json_object(raw))
    if not isinstance(value, dict):
        raise ValueError("tool action is not an object")
    validate_tool_call(value)
    return value


def parse_complete_code(raw: str) -> str | None:
    fences = re.findall(r"```(?:python)?\s*\n(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    candidates = fences or [raw.strip()]
    for code in candidates:
        try:
            compile(code, "<generated>", "exec")
            return code.strip() + "\n"
        except (SyntaxError, ValueError, TypeError):
            continue
    return None

