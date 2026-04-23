"""
Extract failed and underperforming scenarios from eval_metrics.json.

A scenario is considered failed if any scorer falls below --threshold (default 1.0).
Prints a structured report and writes it to a .txt file (default: failures.txt).

Usage:
    python agent-evaluation/extract_failures.py
    python agent-evaluation/extract_failures.py --threshold 0.75
    python agent-evaluation/extract_failures.py --metrics eval_metrics.json --output failures.txt
"""

import argparse
import json
from pathlib import Path

DEFAULT_METRICS_PATH = Path(__file__).resolve().parent / "eval_metrics.json"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "failures.txt"
DEFAULT_THRESHOLD = 1.0


def load_metrics(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(p) as f:
        return json.load(f)


def _fmt_types(label: str, types: list[str], width: int = 12) -> str:
    return f"           {label:<{width}}: {', '.join(types) if types else '(none)'}"


def extract_failures(metrics: dict, threshold: float) -> dict:
    failures_by_scorer: dict[str, list[dict]] = {}
    failed_cases: list[dict] = []

    for i, case in enumerate(metrics.get("per_case", [])):
        if case.get("error"):
            failed_cases.append({
                "case_index": i,
                "category": case.get("category", "unknown"),
                "input_preview": case.get("input_preview", ""),
                "failed_scorers": {"ERROR": case["error"]},
            })
            continue

        failed_scorers = {
            scorer: score
            for scorer, score in case.get("scores", {}).items()
            if score < threshold
        }

        if failed_scorers:
            entry = {
                "case_index": i,
                "category": case.get("category", "unknown"),
                "input_preview": case.get("input_preview", ""),
                "failed_scorers": failed_scorers,
                "all_scores": case.get("scores", {}),
                "scorer_metadata": case.get("scorer_metadata", {}),
            }
            failed_cases.append(entry)
            for scorer in failed_scorers:
                failures_by_scorer.setdefault(scorer, []).append(entry)

    return {
        "threshold": threshold,
        "total_cases": metrics.get("total_cases", 0),
        "failed_case_count": len(failed_cases),
        "failures_by_scorer": {
            scorer: len(cases) for scorer, cases in failures_by_scorer.items()
        },
        "failed_cases": failed_cases,
    }


def print_report(result: dict, file=None) -> None:
    def out(text=""):
        print(text, file=file)

    threshold = result["threshold"]
    total = result["total_cases"]
    failed = result["failed_case_count"]

    out()
    out("=" * 72)
    out(f"FAILURE REPORT  (threshold < {threshold})")
    out("=" * 72)
    out(f"Total cases  : {total}")
    out(f"Failed cases : {failed}  ({failed / total:.1%})" if total else "Failed cases : 0")

    if not result["failures_by_scorer"]:
        out("\nNo failures found at this threshold.")
        out("=" * 72)
        return

    out()
    out("-" * 72)
    out("FAILURES PER SCORER")
    out("-" * 72)
    for scorer, count in sorted(result["failures_by_scorer"].items(), key=lambda x: -x[1]):
        out(f"  {scorer:<40} {count} case(s)")

    out()
    out("-" * 72)
    out("FAILED CASES DETAIL")
    out("-" * 72)
    for case in result["failed_cases"]:
        out(f"\n  [{case['case_index']}] category : {case['category']}")
        out(f"       preview  : {case['input_preview'][:80]}...")
        out("       failures :")
        for scorer, score in case["failed_scorers"].items():
            if isinstance(score, float):
                out(f"         {scorer:<38} score = {score:.4f}")
            else:
                out(f"         {scorer:<38} {score}")
            if scorer == "ExpectedClauseType":
                meta = case.get("scorer_metadata", {}).get("ExpectedClauseType")
                if meta:
                    missing = sorted(set(meta["expected_types"]) - set(meta["matched_types"]))
                    out(_fmt_types("expected", meta["expected_types"]))
                    out(_fmt_types("found   ", meta["found_types"]))
                    out(_fmt_types("missing ", missing))

    out()
    out("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract failed scenarios from eval_metrics.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--metrics", type=str, default=str(DEFAULT_METRICS_PATH),
        help=f"Path to eval_metrics.json (default: {DEFAULT_METRICS_PATH})",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Score threshold — cases below this are reported as failures (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT_PATH),
        help=f"Path to write the failures report as plain text (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    metrics = load_metrics(args.metrics)
    result = extract_failures(metrics, args.threshold)

    print_report(result)

    with open(args.output, "w") as f:
        print_report(result, file=f)
    print(f"Failures written to {args.output}\n")


if __name__ == "__main__":
    main()
