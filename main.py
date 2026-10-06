"""Flight-booking assignment CLI (Python 3.11+, Windows UTF-8)."""
import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        from app.config import ROOT, load_environment, local_today
        from app.evaluation.evaluator import evaluate
        from app.evaluation.scenarios import Scenario, scenarios
        from app.models import ModelConfigurationError, create_model
        from app.runner import run_scenario
        from app.state import Confirmation
    except ModuleNotFoundError as exc:
        print(f"Thiếu dependency {exc.name}. Chạy: python -m pip install -r requirements.txt", file=sys.stderr)
        return 2
    load_environment()
    parser = argparse.ArgumentParser(description="BTVN#3 · đặt vé máy bay MÔ PHỎNG")
    parser.add_argument("--architecture", choices=["react", "plan", "hybrid"], default="react")
    parser.add_argument("--model", choices=["fake", "real"], default="fake")
    parser.add_argument("--today", type=date.fromisoformat, default=local_today(), help="Reference date YYYY-MM-DD, mặc định ngày Việt Nam")
    parser.add_argument("--scenario", default="happy", help="Tên scenario; mặc định fixture đã xác nhận")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json-output", type=Path, help="Lưu state/evaluation trong Flight-Booking-Agent")
    parser.add_argument("--request", help="Yêu cầu riêng; mặc định chưa xác nhận")
    parser.add_argument("--passenger", help="Tên hành khách cung cấp ngoài model")
    parser.add_argument("--confirm", action="store_true", help="Xác nhận mock booking trong phạm vi các --confirm-* dưới đây")
    parser.add_argument("--confirm-origin", default="SGN")
    parser.add_argument("--confirm-destination", default="HAN")
    parser.add_argument("--confirm-date", help="Ngày được xác nhận; mặc định ngày mai")
    parser.add_argument("--confirm-max-price", type=int, default=2_000_000)
    args = parser.parse_args()
    output = args.json_output
    if args.evaluate and output is None:
        output = ROOT / "results" / "evaluation.json"
    if output:
        output = output.resolve()
        if not output.is_relative_to(ROOT.resolve()):
            parser.error("--json-output phải nằm trong Flight-Booking-Agent.")
    try:
        # Validate configuration once before running any scenarios, with no API request.
        create_model(args.model)
        if args.evaluate:
            evaluate(args.today, args.model, output)
            return 0
        fixtures = scenarios(args.today)
        if args.request:
            passenger = args.passenger if args.passenger is not None else ""
            receipt = Confirmation(args.confirm, args.confirm_origin, args.confirm_destination,
                                   args.confirm_date or (args.today + timedelta(days=1)).isoformat(),
                                   args.confirm_max_price, passenger)
            scenario = Scenario("custom", args.request, receipt, "unspecified", passenger=passenger)
        else:
            if args.scenario not in fixtures:
                parser.error(f"Scenario không tồn tại. Chọn: {', '.join(fixtures)}")
            scenario = fixtures[args.scenario]
            if args.passenger is not None:
                scenario = replace(scenario, passenger=args.passenger,
                                   confirmation=replace(scenario.confirmation, passenger_name=args.passenger))
        state = run_scenario(args.architecture, scenario, args.today, args.model, not args.quiet)
    except ModelConfigurationError as exc:
        print(f"CẤU HÌNH MODEL: {exc}", file=sys.stderr)
        return 2
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "status": state.status, "request": state.request.model_dump() if state.request else None,
            "confirmation": asdict(state.confirmation), "summary": state.final_summary,
            "completion_missing": state.completion_missing, "metrics": asdict(state.metrics),
            "observations": state.observations, "attempted_actions": state.attempted_actions,
            "plan_history": state.plan_history, "handoff": state.handoff,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.quiet:
        print(f"{args.architecture}: {state.status}; reason={(state.handoff or {}).get('stop_reason', '')}")
    # Handoff is an expected demo outcome, not a process crash. Config/CLI errors return 2.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
