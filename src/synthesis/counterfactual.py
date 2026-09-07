from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .classify import CounterfactualStats, classify_post_submit, classify_pre_submit
from .generation import TextGenerator, generate_artifact, parse_tool_action
from .normalize import normalized_input_equal
from .prompting.propose_input import build_proposal_prompt, parse_proposed_input, validate_proposal_provenance
from .prompting.repair import build_action_prompt
from .prompting.state_builder import append_tool_observation, build_post_submit_state, build_pre_submit_state
from .schemas import CleanProblem
from .tools import ToolEnvironment


@dataclass(slots=True)
class BranchAttempt:
    seed: int
    action: dict[str, Any] | None
    accepted: bool
    raw_output: str
    messages: list[dict[str, Any]]
    reject_reason: str | None = None


@dataclass(slots=True)
class CounterfactualOutcome:
    behavior: str | None
    stats: CounterfactualStats
    direct: list[BranchAttempt]
    with_run: list[BranchAttempt]
    observation: dict[str, Any] | None
    execution_input: str | None
    proposal: dict[str, Any] | None = None


def _rollout(
    generator: TextGenerator,
    messages: list[dict[str, Any]],
    seed: int,
    environment: ToolEnvironment,
    allow_run: bool,
    limits: tuple[int, int, int] = (6, 3, 3),
    initial_feedback: dict[str, Any] | None = None,
) -> BranchAttempt:
    max_actions, max_submits, max_runs = limits
    state = [dict(message) for message in messages]
    submit_count = 0
    run_count = 0
    latest_feedback = initial_feedback
    raw_outputs: list[str] = []
    last_action: dict[str, Any] | None = None
    for step in range(max_actions):
        artifact = generate_artifact(
            generator, build_action_prompt(state),
            seed + step * 1_000_003,
        )
        raw_outputs.append(artifact.raw_output)
        try:
            action = parse_tool_action(artifact.raw_output)
        except (ValueError, TypeError) as exc:
            return BranchAttempt(
                seed, None, False, "\n".join(raw_outputs), state,
                type(exc).__name__,
            )
        last_action = action
        if action["name"] == "run_candidate":
            if not allow_run or run_count >= max_runs:
                return BranchAttempt(
                    seed, action, False, "\n".join(raw_outputs), state,
                    "run_not_allowed_or_budget_exhausted",
                )
            if latest_feedback is not None:
                failing = latest_feedback.get("failing_input")
                if not isinstance(failing, str) or not normalized_input_equal(
                    action["arguments"]["input"], failing,
                    environment.problem.public_problem.io_mode,
                ):
                    return BranchAttempt(
                        seed, action, False, "\n".join(raw_outputs),
                        state, "run_input_not_latest_failing_case",
                    )
            run_count += 1
            observation = environment.call(
                "run_candidate", action["arguments"],
            )
            state = append_tool_observation(
                state, "run_candidate", action["arguments"],
                observation, trainable=True,
            )
            continue
        if submit_count >= max_submits:
            return BranchAttempt(
                seed, action, False, "\n".join(raw_outputs), state,
                "submit_budget_exhausted",
            )
        submit_count += 1
        result = environment.call("submit", action["arguments"])
        accepted = (
            result["status"] == "accepted"
            and result["pass_rate"] == 1.0
        )
        if accepted:
            replay = environment.call("submit", action["arguments"])
            accepted = (
                replay["status"] == "accepted"
                and replay["pass_rate"] == 1.0
            )
            state = append_tool_observation(
                state, "submit", action["arguments"], result,
                trainable=accepted,
            )
            return BranchAttempt(
                seed, action, accepted, "\n".join(raw_outputs), state,
                None if accepted else "accepted_replay_failed",
            )
        state = append_tool_observation(
            state, "submit", action["arguments"], result,
            trainable=False,
        )
        latest_feedback = result
    return BranchAttempt(
        seed, last_action, False, "\n".join(raw_outputs), state,
        "action_budget_exhausted",
    )


def evaluate_post_submit(
    problem: CleanProblem,
    candidate: str,
    seed_feedback: dict[str, Any],
    generator: TextGenerator,
    seeds: list[int],
    environment_factory,
    thresholds=(2 / 3, 1 / 3, 1 / 3),
    limits=(6, 3, 3),
) -> CounterfactualOutcome:
    state = build_post_submit_state(
        problem.public_problem, candidate, seed_feedback,
    )
    direct = [
        _rollout(
            generator, state, seed,
            environment_factory(problem, candidate),
            False, limits, seed_feedback,
        )
        for seed in seeds
    ]
    failing = seed_feedback.get("failing_input")
    with_run: list[BranchAttempt] = []
    observation = None
    if isinstance(failing, str):
        run_env = environment_factory(problem, candidate)
        observation = run_env.call(
            "run_candidate", {"input": failing},
        )
        run_state = append_tool_observation(
            state, "run_candidate", {"input": failing}, observation,
        )
        remaining = (
            max(0, limits[0] - 1),
            limits[1],
            max(0, limits[2] - 1),
        )
        if observation["status"] != "output_limit":
            with_run = [
                _rollout(
                    generator, run_state, seed,
                    environment_factory(problem, candidate),
                    True, remaining, seed_feedback,
                )
                for seed in seeds
            ]
    high, low, minimum = thresholds
    stats = CounterfactualStats(
        len(seeds),
        sum(item.accepted for item in direct),
        sum(item.accepted for item in with_run),
        seeds, high, low, minimum,
    )
    matches = (
        isinstance(failing, str)
        and normalized_input_equal(
            failing, failing, problem.public_problem.io_mode,
        )
    )
    behavior = classify_post_submit(
        stats, isinstance(failing, str), matches,
    )
    return CounterfactualOutcome(
        behavior, stats, direct, with_run, observation,
        failing if isinstance(failing, str) else None,
    )


def evaluate_pre_submit(
    problem: CleanProblem,
    candidate: str,
    generator: TextGenerator,
    proposal_seed: int,
    seeds: list[int],
    environment_factory,
    thresholds=(2 / 3, 1 / 3, 1 / 3),
    limits=(6, 3, 3),
) -> CounterfactualOutcome:
    state = build_pre_submit_state(problem.public_problem, candidate)
    direct = [
        _rollout(
            generator, state, seed,
            environment_factory(problem, candidate),
            False, limits,
        )
        for seed in seeds
    ]
    prompt, fields = build_proposal_prompt(
        problem.public_problem, candidate,
    )
    validate_proposal_provenance(fields)
    proposal_artifact = generate_artifact(
        generator, prompt, proposal_seed,
    )
    with_run: list[BranchAttempt] = []
    observation = None
    proposed_input = None
    executable = False
    try:
        proposed_input = parse_proposed_input(
            proposal_artifact.raw_output,
        )
        run_env = environment_factory(problem, candidate)
        observation = run_env.call(
            "run_candidate", {"input": proposed_input},
        )
        executable = observation["status"] not in {
            "invalid_input", "timeout", "output_limit",
        }
        run_state = append_tool_observation(
            state, "run_candidate",
            {"input": proposed_input}, observation,
        )
        remaining = (
            max(0, limits[0] - 1),
            limits[1],
            max(0, limits[2] - 1),
        )
        with_run = [
            _rollout(
                generator, run_state, seed,
                environment_factory(problem, candidate),
                True, remaining,
            )
            for seed in seeds
        ]
    except (ValueError, TypeError):
        pass
    high, low, minimum = thresholds
    stats = CounterfactualStats(
        len(seeds),
        sum(item.accepted for item in direct),
        sum(item.accepted for item in with_run),
        seeds, high, low, minimum,
    )
    behavior = classify_pre_submit(stats, True, executable)
    proposal = {
        "model": proposal_artifact.model,
        "revision": proposal_artifact.revision,
        "seed": proposal_seed,
        "prompt_hash": proposal_artifact.prompt_hash,
        "raw_hash": proposal_artifact.raw_hash,
        "observable_fields": list(fields),
    }
    return CounterfactualOutcome(
        behavior, stats, direct, with_run, observation,
        proposed_input, proposal,
    )

