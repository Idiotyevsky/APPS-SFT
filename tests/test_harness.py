from synthesis.grader import PrivateGrader
from synthesis.sandbox import SandboxConfig, SandboxedExecutor


def grader(output=65536):
    config = SandboxConfig(timeout_sec=1.5, max_output_bytes=output)
    return PrivateGrader(SandboxedExecutor(config))


def test_stdin_grader_counts_apps_items_not_inner_lines():
    code = "t=int(input())\nfor _ in range(t): print(int(input())*2)\n"
    result = grader().grade(
        code,
        ["2\n2\n3\n"],
        ["4\n6\n"],
        "stdin",
    )
    assert result.accepted
    assert result.total == 1


def test_call_mode_global_function_and_solution_class():
    direct = "def add(a, b):\n    return a + b\n"
    result = grader().grade(
        direct, [[2, 3]], [5], "call", "add",
    )
    assert result.accepted
    method = (
        "class Solution:\n"
        "    def add(self, a, b):\n"
        "        return a + b\n"
    )
    result = grader().grade(
        method, [[2, 3]], [5], "call", "add",
    )
    assert result.accepted


def test_timeout_and_output_limit():
    timeout = grader().grade(
        "while True: pass\n", ["x"], [""], "stdin",
    )
    assert timeout.status == "timeout"
    output = grader(output=4).grade(
        "print('abcdefgh')\n", [""], ["abcdefgh\n"], "stdin",
    )
    assert output.status == "runtime_error"
    assert output.cases[0].run.truncated
def test_submit_compile_error_has_no_failing_input():
    result = grader().grade(
        "def broken(:\n", ["1\n"], ["1\n"], "stdin",
    )
    assert result.status == "compile_error"
    assert result.failing_input is None
    assert result.total == 1


def test_call_mode_rejects_non_json_argument_list():
    run, actual = grader().run_candidate(
        "def f(x): return x\n", '{"x":1}', "call", "f",
    )
    assert run.status == "invalid_input"
    assert actual is None
