"""Measure actual graph runs; persist both metrics and auditable tool traces."""
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from statistics import mean

from app.evaluation.scenarios import Scenario, scenarios
from app.runner import BUILDERS, run_scenario


def expectation_pass(architecture: str, scenario: Scenario, status: str, reason: str) -> bool:
    if scenario.expected_status == "architecture_dependent":
        return (status == "done") if architecture == "plan" else (
            status == "handoff" and reason == "model_budget_exceeded"
        )
    return status == scenario.expected_status and (not scenario.expected_reason or reason.startswith(scenario.expected_reason))


def evaluate(today: date, mode: str = "fake", output: Path | None = None) -> dict:
    rows = []
    for architecture in BUILDERS:
        for scenario in scenarios(today).values():
            state = run_scenario(architecture, scenario, today, mode, verbose=False)
            reason = (state.handoff or {}).get("stop_reason", "")
            row = {
                "architecture": architecture, "scenario": scenario.name,
                "success": state.status == "done", "completion_valid": state.status == "done" and not state.completion_missing,
                "status": state.status, "stop_reason": reason,
                "scenario_pass": expectation_pass(architecture, scenario, state.status, reason) if mode == "fake" else None,
                **asdict(state.metrics),
                "approximate_cost": None,
                "token_metric": (state.metrics.input_tokens + state.metrics.output_tokens) if state.metrics.token_usage_reported else None,
                "booking": state.booking,
                "attempted_actions": state.attempted_actions, "observations": state.observations,
                "plan_history": state.plan_history, "handoff": state.handoff,
            }
            rows.append(row)
            print(f"{architecture:6} | {scenario.name:20} | {state.status:7} | model={state.metrics.model_calls:2} tool={state.metrics.tool_calls:2} | {reason}")
    aggregates = []
    for architecture in BUILDERS:
        group = [r for r in rows if r["architecture"] == architecture]
        aggregates.append({
            "architecture": architecture, "runs": len(group),
            "success": sum(r["success"] for r in group),
            "completion_valid": sum(r["completion_valid"] for r in group),
            "scenario_pass": sum(r["scenario_pass"] is True for r in group) if mode == "fake" else None,
            **{key: round(mean(r[key] for r in group), 3) for key in ("model_calls", "tool_calls", "total_steps", "elapsed_seconds")},
            **{key: sum(r[key] for r in group) for key in ("blocked_actions", "handoff_count", "replans", "loop_detected", "stall_detected")},
        })
    print("\nArchitecture | Success | Valid | Model(avg) | Tool(avg) | Steps(avg) | Blocked | Handoff | Replans")
    print("-" * 101)
    for row in aggregates:
        print(f"{row['architecture']:12} | {row['success']:2}/{row['runs']:<2}   | {row['completion_valid']:5} | {row['model_calls']:10.3f} | {row['tool_calls']:9.3f} | {row['total_steps']:10.3f} | {row['blocked_actions']:7} | {row['handoff_count']:7} | {row['replans']:7}")
    print("\nSuccess = booking DONE, không tính handoff là đặt vé thành công. Scenario pass = đạt kết cục an toàn kỳ vọng.")
    print("ReAct phản ứng từng observation; Plan ít model calls nhưng phải replan khi lỗi; Hybrid thêm chi phí kiểm stage.")
    print("Fake cost/token = N/A. Elapsed phụ thuộc máy; không suy ra chất lượng LLM từ model giả.")
    result = {"reference_date": today.isoformat(), "model": mode, "aggregates": aggregates, "runs": rows}
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        output.with_suffix(".md").write_text(format_markdown(result), encoding="utf-8")
        print(f"Saved: {output}")
    return result


def format_markdown(result: dict) -> str:
    """This artifact is computed from runtime records, never manually invented."""
    lines = ["# Kết quả đánh giá runtime", "",
             f"Ngày tham chiếu: {result['reference_date']}; model: `{result['model']}`.", "",
             "Model/tool/steps là trung bình mỗi scenario; blocked/handoff/replans là tổng.", "",
             "| Architecture | Success | Valid | Scenario pass | Model avg | Tool avg | Steps avg | Blocked | Handoff | Replans |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["aggregates"]:
        lines.append(f"| {row['architecture']} | {row['success']}/{row['runs']} | {row['completion_valid']} | "
                     f"{row['scenario_pass']} | {row['model_calls']:.3f} | {row['tool_calls']:.3f} | "
                     f"{row['total_steps']:.3f} | {row['blocked_actions']} | {row['handoff_count']} | {row['replans']} |")
    lines += ["", "| Architecture | Scenario | Status | Calls model/tool | Steps | Reason |",
              "|---|---|---|---:|---:|---|"]
    for row in result["runs"]:
        lines.append(f"| {row['architecture']} | {row['scenario']} | {row['status']} | "
                     f"{row['model_calls']}/{row['tool_calls']} | {row['total_steps']} | {row['stop_reason']} |")
    lines += ["", "Success chỉ đếm booking được CompletionChecker chứng minh. Handoff an toàn không là success.",
              "Fake cost/token = N/A; real token chỉ có khi provider cung cấp usage_metadata.",
              "Elapsed và trace đầy đủ nằm trong JSON cùng tên. Kết quả fake không chứng minh chất lượng LLM thật.", ""]
    return "\n".join(lines)
