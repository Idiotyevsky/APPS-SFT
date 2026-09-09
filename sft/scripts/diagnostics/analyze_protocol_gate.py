#!/usr/bin/env python3
"""Compare ToolAPPS agent-gate trajectories without changing evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _next_tool(events: list[dict], index: int) -> str | None:
    for event in events[index + 1:]:
        if event.get("tool"):
            return event["tool"]
    return None


def analyze(path: Path) -> dict:
    payload = json.loads(path.read_text())
    rows = payload["by_problem"]
    counts = Counter()
    status_counts = Counter()

    for row in rows:
        events = row.get("turn_history", [])
        tools = [(index, event) for index, event in enumerate(events)
                 if event.get("tool")]
        submits = [(index, event) for index, event in tools
                   if event["tool"] == "submit"]
        status_counts.update([row.get("status", "unknown")])

        if row.get("duplicate_submissions", 0):
            counts["problems_with_duplicate_submission"] += 1
        if row.get("status") == "action_budget" and tools and tools[-1][1]["tool"] == "run_candidate":
            counts["action_budget_ending_in_run"] += 1

        if submits:
            counts["first_submit_attempts"] += 1
            first_index, first = submits[0]
            first_observation = first.get("observation", {})
            if first_observation.get("status") != "compile_error":
                counts["first_submit_compile_valid"] += 1
            if first_observation.get("status") != "accepted":
                counts["post_first_failure_total"] += 1
                if _next_tool(events, first_index) == "run_candidate":
                    counts["post_first_failure_run_count"] += 1

        previous_code = None
        previous_pass_rate = None
        submit_number = 0
        for index, event in tools:
            if event["tool"] != "submit":
                continue
            submit_number += 1
            action = event.get("action", {})
            code = action.get("arguments", {}).get("code")
            observation = event.get("observation", {})
            pass_rate = observation.get("pass_rate")
            if previous_code is not None:
                counts["submit_transitions"] += 1
                if code == previous_code:
                    counts["identical_submit_transitions"] += 1
                else:
                    counts["changed_submit_transitions"] += 1
                if isinstance(pass_rate, (int, float)) and isinstance(previous_pass_rate, (int, float)):
                    if pass_rate > previous_pass_rate:
                        counts["pass_rate_improved"] += 1
                    elif pass_rate < previous_pass_rate:
                        counts["pass_rate_worsened"] += 1
                    else:
                        counts["pass_rate_equal"] += 1
            if submit_number == 2 and observation.get("status") != "accepted":
                counts["second_failure_cases"] += 1
                if _next_tool(events, index) == "run_candidate":
                    counts["second_failure_run_count"] += 1
            previous_code = code
            previous_pass_rate = pass_rate

        current_code = None
        for index, event in tools:
            if event["tool"] == "submit":
                current_code = event.get("action", {}).get("arguments", {}).get("code")
                continue
            counts["run_events"] += 1
            following = next((candidate for candidate_index, candidate in tools
                              if candidate_index > index and candidate["tool"] == "submit"), None)
            if following is None:
                counts["run_without_followup_submit"] += 1
                continue
            counts["run_with_followup_submit"] += 1
            following_code = following.get("action", {}).get("arguments", {}).get("code")
            if current_code is not None and following_code != current_code:
                counts["run_followed_by_changed_submit"] += 1

    summary = payload["summary"]
    derived = dict(counts)
    derived.update({
        "first_submit_compile_valid_rate": _rate(
            counts["first_submit_compile_valid"], counts["first_submit_attempts"]),
        "post_first_failure_run_rate": _rate(
            counts["post_first_failure_run_count"], counts["post_first_failure_total"]),
        "second_failure_run_rate": _rate(
            counts["second_failure_run_count"], counts["second_failure_cases"]),
        "changed_submit_transition_rate": _rate(
            counts["changed_submit_transitions"], counts["submit_transitions"]),
        "pass_rate_improvement_rate": _rate(
            counts["pass_rate_improved"], counts["submit_transitions"]),
        "run_followed_by_changed_submit_rate": _rate(
            counts["run_followed_by_changed_submit"], counts["run_with_followup_submit"]),
    })
    return {
        "path": str(path.resolve()),
        "summary": summary,
        "terminal_status": dict(status_counts),
        "derived": derived,
    }


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", required=True,
                        help="LABEL=path/to/results.json")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    comparison = {}
    for item in args.result:
        label, separator, path = item.partition("=")
        if not separator or not label or not path:
            parser.error(f"invalid --result: {item!r}")
        comparison[label] = analyze(Path(path))

    columns = [
        "model", "solved", "success", "repair", "format", "mean_runs",
        "duplicates", "compile_valid", "post_fail_run", "changed_submit",
        "pass_improved", "run_to_change", "wasted_final_run",
    ]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |"]
    for label, data in comparison.items():
        summary, derived = data["summary"], data["derived"]
        values = [
            label,
            f"{summary['solved']}/{summary['completed']}",
            summary["budgeted_agent_success_rate"],
            summary["repair_success_rate"],
            summary["tool_format_valid_rate"],
            summary["mean_runs"],
            summary["duplicate_submissions"],
            derived["first_submit_compile_valid_rate"],
            derived["post_first_failure_run_rate"],
            derived["changed_submit_transition_rate"],
            derived["pass_rate_improvement_rate"],
            derived["run_followed_by_changed_submit_rate"],
            derived.get("action_budget_ending_in_run", 0),
        ]
        lines.append("| " + " | ".join(_fmt(value) for value in values) + " |")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    args.output_md.write_text("# Protocol Gate Comparison\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
