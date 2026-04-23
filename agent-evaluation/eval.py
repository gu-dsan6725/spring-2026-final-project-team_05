"""
Braintrust Evaluations for the Contract Analysis Pipeline.

Evaluates the full LangGraph pipeline (ingestion → classification → risk_analysis →
benchmark → report) using:
- Braintrust Eval() framework
- Autoevals LLM-as-judge scorers (Factuality, ClosedQA) via Claude Sonnet 4.6
- Custom heuristic scorers for each agent stage's output validity

Usage:
    python agent-evaluation/eval.py
    python agent-evaluation/eval.py --dataset eval_dataset.json --output eval_metrics.json
    python agent-evaluation/eval.py --no-send-logs
    python agent-evaluation/eval.py --debug
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Allow imports from the project root (agents/, etc.)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from autoevals.llm import Factuality
from braintrust import Eval
from dotenv import load_dotenv
from openai import OpenAI


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

os.chdir(_PROJECT_ROOT)
load_dotenv()

_EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = str(_EVAL_DIR / "eval_dataset.json")
DEFAULT_OUTPUT_PATH = str(_EVAL_DIR / "eval_metrics.json")
BRAINTRUST_PROJECT_NAME = os.environ.get("BRAINTRUST_PROJECT", "contract-pipeline-evals")

# Side-channel cache populated by the wrapped scorer so _export_eval_metrics
# can persist expected/found/matched type breakdowns to eval_metrics.json.
_clause_type_metadata: dict[str, dict] = {}
EVAL_JUDGE_MODEL = "claude-sonnet-4-6"
ANTHROPIC_OPENAI_BASE_URL = "https://api.anthropic.com/v1/"


def _create_judge_client() -> OpenAI:
    """
    Create an OpenAI-compatible client pointing at Anthropic's API.

    Autoevals scorers use the OpenAI SDK interface; Anthropic's compatible
    endpoint lets us use Claude Sonnet 4.6 as the judge model.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")
    return OpenAI(api_key=api_key, base_url=ANTHROPIC_OPENAI_BASE_URL)


def _load_dataset(dataset_path: str) -> list[dict]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with open(path) as f:
        dataset = json.load(f)
    logger.info(f"Loaded {len(dataset)} test cases from {dataset_path}")
    return dataset


def _run_pipeline_on_input(contract_text: str) -> dict:
    """
    Run the full LangGraph pipeline on a contract and return all intermediate
    state plus timing information.
    """
    from agents.orchestrator_agent import run_pipeline

    logger.info(f"Running pipeline on contract ({len(contract_text)} chars)...")
    start = time.time()

    try:
        result = run_pipeline(contract_text)
        elapsed = time.time() - start
        report = json.loads(result["report"])
    except Exception as exc:
        elapsed = time.time() - start
        logger.error(f"Pipeline failed: {exc}")
        return {
            "pipeline_error": str(exc),
            "report": None,
            "clauses": [],
            "classified_clauses": [],
            "risk_scores": [],
            "benchmark_results": [],
            "latency_seconds": elapsed,
        }

    return {
        "pipeline_error": None,
        "report": report,
        "clauses": result.get("clauses", []),
        "classified_clauses": result.get("classified_clauses", []),
        "risk_scores": result.get("risk_scores", []),
        "benchmark_results": result.get("benchmark_results", []),
        "latency_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Custom Scorers
# ---------------------------------------------------------------------------


def clause_structure_validity_scorer(
    input: str,
    output: str,
    expected: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[dict]:
    """
    Ingestion agent: all parsed clauses must have non-empty id, text, and section.
    """
    if not metadata:
        return None

    clauses = metadata.get("clauses", [])
    if not clauses:
        return {
            "name": "ClauseStructureValidity",
            "score": 0.0,
            "metadata": {"reason": "no clauses produced by ingestion agent"},
        }

    required = {"id", "text", "section"}
    valid = sum(
        1 for c in clauses
        if required.issubset(c.keys()) and str(c.get("text", "")).strip()
    )
    score = valid / len(clauses)

    return {
        "name": "ClauseStructureValidity",
        "score": score,
        "metadata": {"total_clauses": len(clauses), "valid_clauses": valid},
    }


def classification_validity_scorer(
    input: str,
    output: str,
    expected: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[dict]:
    """
    Classification agent: each clause must have a non-empty clause_type and a
    confidence value in [0, 1].
    """
    if not metadata:
        return None

    classified = metadata.get("classified_clauses", [])
    if not classified:
        return {
            "name": "ClassificationValidity",
            "score": 0.0,
            "metadata": {"reason": "no classified clauses"},
        }

    valid = sum(
        1 for c in classified
        if c.get("clause_type", "") and 0.0 <= c.get("confidence", -1) <= 1.0
    )
    score = valid / len(classified)

    return {
        "name": "ClassificationValidity",
        "score": score,
        "metadata": {"total": len(classified), "valid": valid},
    }


def expected_clause_type_scorer(
    input: str,
    output: str,
    expected: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[dict]:
    """
    Classification agent: fraction of expected CUAD clause types found in
    the classified output. Penalises missing types; does not penalise extras.
    """
    if not metadata:
        return None

    expected_types = metadata.get("expected_clause_types", [])
    classified = metadata.get("classified_clauses", [])
    if not expected_types or not classified:
        return None

    found = {c.get("clause_type", "") for c in classified}
    expected_set = set(expected_types)
    matched = expected_set & found
    score = len(matched) / len(expected_set)

    return {
        "name": "ExpectedClauseType",
        "score": score,
        "metadata": {
            "expected_types": sorted(expected_set),
            "found_types": sorted(found),
            "matched_types": sorted(matched),
        },
    }


def _expected_clause_type_scorer_with_cache(
    input: str,
    output: str,
    expected: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[dict]:
    result = expected_clause_type_scorer(input, output, expected, metadata)
    if result and result.get("metadata"):
        _clause_type_metadata[input] = result["metadata"]
    return result


def risk_score_validity_scorer(
    input: str,
    output: str,
    expected: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[dict]:
    """
    Risk analysis agent: risk_score must be in [0, 1] and risk_factors must be a list.
    """
    if not metadata:
        return None

    risk_scores = metadata.get("risk_scores", [])
    if not risk_scores:
        return {
            "name": "RiskScoreValidity",
            "score": 0.0,
            "metadata": {"reason": "no risk scores produced"},
        }

    valid = sum(
        1 for c in risk_scores
        if 0.0 <= c.get("risk_score", -1) <= 1.0
        and isinstance(c.get("risk_factors"), list)
    )
    score = valid / len(risk_scores)

    return {
        "name": "RiskScoreValidity",
        "score": score,
        "metadata": {"total": len(risk_scores), "valid": valid},
    }


def risk_factors_presence_scorer(
    input: str,
    output: str,
    expected: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[dict]:
    """
    Risk analysis agent: high-risk clauses (score >= 0.5) should have at least one
    identified risk factor, not an empty list.
    """
    if not metadata:
        return None

    risk_scores = metadata.get("risk_scores", [])
    high_risk = [c for c in risk_scores if c.get("risk_score", 0) >= 0.5]
    if not high_risk:
        return None

    with_factors = sum(1 for c in high_risk if c.get("risk_factors", []))
    score = with_factors / len(high_risk)

    return {
        "name": "RiskFactorsPresence",
        "score": score,
        "metadata": {"high_risk_clauses": len(high_risk), "with_factors": with_factors},
    }


def benchmark_similarity_validity_scorer(
    input: str,
    output: str,
    expected: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[dict]:
    """
    Benchmark agent: benchmark_similarity must be in [0, 1] and a source must be
    provided for every clause.
    """
    if not metadata:
        return None

    benchmark = metadata.get("benchmark_results", [])
    if not benchmark:
        return {
            "name": "BenchmarkSimilarityValidity",
            "score": 0.0,
            "metadata": {"reason": "no benchmark results produced"},
        }

    valid = sum(
        1 for c in benchmark
        if 0.0 <= c.get("benchmark_similarity", -1) <= 1.0
        and c.get("benchmark_source", "")
    )
    score = valid / len(benchmark)

    return {
        "name": "BenchmarkSimilarityValidity",
        "score": score,
        "metadata": {"total": len(benchmark), "valid": valid},
    }


def output_structure_validity_scorer(
    input: str,
    output: str,
    expected: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[dict]:
    """
    Report node: the final JSON report must contain a summary with total_clauses,
    a non-empty clauses list, and each clause must have the required keys.
    """
    if not metadata:
        return None

    report = metadata.get("report")
    if report is None:
        return {
            "name": "OutputStructureValidity",
            "score": 0.0,
            "metadata": {"reason": "no report produced"},
        }

    checks = {
        "has_summary": "summary" in report,
        "has_clauses_key": "clauses" in report,
        "has_total_clauses": "total_clauses" in report.get("summary", {}),
        "clauses_is_list": isinstance(report.get("clauses"), list),
        "clauses_not_empty": len(report.get("clauses", [])) > 0,
    }

    required_clause_keys = {"id", "clause_type", "risk_score", "benchmark_similarity"}
    checks["clause_keys_valid"] = all(
        required_clause_keys.issubset(c.keys())
        for c in report.get("clauses", [{}])
    )

    score = sum(checks.values()) / len(checks)

    return {
        "name": "OutputStructureValidity",
        "score": score,
        "metadata": checks,
    }


def latency_scorer(
    input: str,
    output: str,
    expected: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[dict]:
    """
    Pipeline latency scorer. Thresholds are tuned for a multi-agent pipeline
    that makes ~3 sequential LLM calls per clause.

    < 30s  → 1.0   (fast)
    30–60s → 0.75
    60–120s→ 0.5
    120–180s→ 0.25
    > 180s → 0.0   (too slow)
    """
    if not metadata:
        return None

    latency = metadata.get("latency_seconds")
    if latency is None:
        return None

    if latency < 30:
        score = 1.0
    elif latency < 60:
        score = 0.75
    elif latency < 120:
        score = 0.5
    elif latency < 180:
        score = 0.25
    else:
        score = 0.0

    return {
        "name": "Latency",
        "score": score,
        "metadata": {"latency_seconds": round(latency, 2)},
    }


def no_error_scorer(
    input: str,
    output: str,
    expected: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[dict]:
    """
    Pipeline-level: score 1.0 if the pipeline completed without raising an
    exception and produced a non-None report.
    """
    if not metadata:
        return None

    error = metadata.get("pipeline_error")
    if error:
        return {
            "name": "NoError",
            "score": 0.0,
            "metadata": {"error": str(error)[:200]},
        }

    if metadata.get("report") is None:
        return {
            "name": "NoError",
            "score": 0.0,
            "metadata": {"reason": "no report produced"},
        }

    return {"name": "NoError", "score": 1.0, "metadata": {}}


# ---------------------------------------------------------------------------
# Task Function and Data Loader
# ---------------------------------------------------------------------------


def _create_wrapped_task(dataset: list[dict]):
    """
    Run the pipeline inside data() and cache results so that runtime metadata
    (per-agent outputs, latency) is available to scorers via the metadata dict.

    This mirrors the lab10 pattern: Braintrust passes metadata from data() to
    every scorer, but pipeline outputs are only known at runtime — so we run
    the pipeline here and inject the results into metadata.
    """
    results_cache: dict[str, dict] = {}

    def data() -> list[dict]:
        cases = []
        for case in dataset:
            contract_text = case["input"]
            category = case.get("category", "unknown")
            logger.info(f"Running pipeline for test case: [{category}]")

            result = _run_pipeline_on_input(contract_text)
            results_cache[contract_text] = result

            cases.append({
                "input": contract_text,
                "expected": case.get("expected_output", ""),
                "metadata": {
                    "category": case.get("category", ""),
                    "difficulty": case.get("difficulty", ""),
                    "expected_clause_types": case.get("expected_clause_types", []),
                    # Runtime outputs injected for custom scorers
                    "clauses": result.get("clauses", []),
                    "classified_clauses": result.get("classified_clauses", []),
                    "risk_scores": result.get("risk_scores", []),
                    "benchmark_results": result.get("benchmark_results", []),
                    "report": result.get("report"),
                    "pipeline_error": result.get("pipeline_error"),
                    "latency_seconds": result.get("latency_seconds"),
                },
            })

        return cases

    def task(input: str) -> str:
        # Pipeline already ran in data(); return the cached report JSON.
        if input in results_cache:
            report = results_cache[input].get("report")
            return json.dumps(report, indent=2) if report else "Pipeline failed — no report produced"
        # Fallback if cache miss (should not happen in normal eval flow)
        result = _run_pipeline_on_input(input)
        report = result.get("report")
        return json.dumps(report, indent=2) if report else "Pipeline failed — no report produced"

    return task, data


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_eval_summary(eval_result: Any, dataset: list[dict]) -> None:
    results = eval_result.results
    if not results:
        logger.warning("No evaluation results to summarize")
        return

    category_lookup = {case["input"]: case.get("category", "unknown") for case in dataset}

    scorer_scores: dict[str, list[float]] = {}
    category_scores: dict[str, list[float]] = {}
    error_cases = []

    for r in results:
        input_text = str(r.input) if r.input else ""
        category = category_lookup.get(input_text, "unknown")

        if r.error:
            error_cases.append({"input": input_text[:80], "error": str(r.error)})
            continue

        for scorer_name, score_val in r.scores.items():
            if score_val is None:
                continue
            scorer_scores.setdefault(scorer_name, []).append(score_val)
            category_scores.setdefault(f"{category}/{scorer_name}", []).append(score_val)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY — CONTRACT ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Total test cases : {len(results)}")
    print(f"Errors           : {len(error_cases)}")
    print()

    print("-" * 80)
    print(f"{'Scorer':<35} {'Avg':>10} {'Min':>8} {'Max':>8} {'N':>6}")
    print("-" * 80)
    for name in sorted(scorer_scores):
        scores = scorer_scores[name]
        avg = sum(scores) / len(scores)
        print(f"{name:<35} {avg:>10.2%} {min(scores):>8.2f} {max(scores):>8.2f} {len(scores):>6}")

    print()
    print("-" * 80)
    print("PER-CATEGORY BREAKDOWN")
    print("-" * 80)
    categories = sorted({case.get("category", "unknown") for case in dataset})
    for cat in categories:
        print(f"\n  [{cat}]")
        for name in sorted(scorer_scores):
            key = f"{cat}/{name}"
            if key in category_scores:
                scores = category_scores[key]
                print(f"    {name:<33} {sum(scores)/len(scores):>8.2%}  (n={len(scores)})")

    if error_cases:
        print()
        print("-" * 80)
        print("FAILED CASES")
        print("-" * 80)
        for case in error_cases:
            print(f"  Input: {case['input']}")
            print(f"  Error: {case['error']}")
            print()

    print("=" * 80 + "\n")
    logger.info(
        f"Eval summary: {len(results)} cases, {len(error_cases)} errors — "
        + ", ".join(
            f"{k}={sum(v)/len(v):.2%}"
            for k, v in sorted(scorer_scores.items())
        )
    )


def _export_eval_metrics(
    eval_result: Any,
    dataset: list[dict],
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> None:
    results = eval_result.results
    if not results:
        logger.warning("No results to export")
        return

    category_lookup = {case["input"]: case.get("category", "unknown") for case in dataset}

    scorer_scores: dict[str, list[float]] = {}
    category_scores: dict[str, list[float]] = {}
    per_case_results = []
    error_count = 0

    for r in results:
        input_text = str(r.input) if r.input else ""
        category = category_lookup.get(input_text, "unknown")

        clause_type_meta = _clause_type_metadata.get(input_text)
        case_entry: dict = {
            "input_preview": input_text[:120],
            "category": category,
            "scores": {},
            "scorer_metadata": {
                "ExpectedClauseType": clause_type_meta,
            } if clause_type_meta else {},
            "error": None,
        }

        if r.error:
            error_count += 1
            case_entry["error"] = str(r.error)
            per_case_results.append(case_entry)
            continue

        for scorer_name, score_val in r.scores.items():
            if score_val is None:
                continue
            case_entry["scores"][scorer_name] = round(score_val, 4)
            scorer_scores.setdefault(scorer_name, []).append(score_val)
            category_scores.setdefault(f"{category}/{scorer_name}", []).append(score_val)

        per_case_results.append(case_entry)

    overall = {
        name: {
            "average": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "count": len(scores),
        }
        for name, scores in sorted(scorer_scores.items())
    }

    categories = sorted({case.get("category", "unknown") for case in dataset})
    per_category: dict[str, dict] = {}
    for cat in categories:
        per_category[cat] = {}
        for name in sorted(scorer_scores):
            key = f"{cat}/{name}"
            if key in category_scores:
                scores = category_scores[key]
                per_category[cat][name] = {
                    "average": round(sum(scores) / len(scores), 4),
                    "count": len(scores),
                }

    metrics = {
        "total_cases": len(results),
        "errors": error_count,
        "overall_scores": overall,
        "per_category": per_category,
        "per_case": per_case_results,
    }

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Evaluation metrics exported to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Braintrust evaluations on the Contract Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python agent-evaluation/eval.py
    python agent-evaluation/eval.py --dataset eval_dataset.json --output eval_metrics.json
    python agent-evaluation/eval.py --no-send-logs
    python agent-evaluation/eval.py --debug
""",
    )
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET_PATH,
        help=f"Path to evaluation dataset JSON (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_PATH,
        help=f"Path for output eval metrics JSON (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--no-send-logs", action="store_true",
        help="Run evaluations locally without sending results to Braintrust",
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help="Name for this evaluation experiment (default: auto-generated)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting Contract Analysis Pipeline Evaluations")
    start = time.time()

    dataset = _load_dataset(args.dataset)
    task_fn, data_fn = _create_wrapped_task(dataset)

    # Use Claude Sonnet 4.6 as the judge model via Anthropic's OpenAI-compatible endpoint
    judge_client = _create_judge_client()

    # Scorers
    # LLM-as-judge (Factuality, ClosedQA) — assess overall report quality vs expected
    # Heuristic per-agent scorers — validate structure and field validity at each stage
    all_scorers = [
        Factuality(model=EVAL_JUDGE_MODEL, client=judge_client),
        clause_structure_validity_scorer,       # ingestion agent
        classification_validity_scorer,         # classification agent
        _expected_clause_type_scorer_with_cache,  # classification agent
        risk_score_validity_scorer,             # risk analysis agent
        risk_factors_presence_scorer,           # risk analysis agent
        benchmark_similarity_validity_scorer,   # benchmark agent
        output_structure_validity_scorer,       # report node
        latency_scorer,                         # pipeline-level
        no_error_scorer,                        # pipeline-level
    ]

    eval_kwargs: dict[str, Any] = {
        "data": data_fn,
        "task": task_fn,
        "scores": all_scorers,
    }

    if args.experiment_name:
        eval_kwargs["experiment_name"] = args.experiment_name

    if args.no_send_logs:
        eval_kwargs["no_send_logs"] = True
        logger.info("Running in local mode (no logs sent to Braintrust)")

    logger.info("Running Braintrust evaluation...")
    eval_result = Eval(BRAINTRUST_PROJECT_NAME, **eval_kwargs)

    _print_eval_summary(eval_result, dataset)
    _export_eval_metrics(eval_result, dataset, output_path=args.output)

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    if minutes > 0:
        logger.info(f"Evaluation completed in {minutes}m {seconds:.1f}s")
    else:
        logger.info(f"Evaluation completed in {seconds:.1f}s")


if __name__ == "__main__":
    main()
