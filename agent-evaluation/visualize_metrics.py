"""
Visualize eval_metrics.json results. Charts are saved to the charts/ subdirectory.

Produces:
  charts/overall_scores.png         — bar chart of avg score per scorer
  charts/category_heatmap.png       — heatmap of avg score by category × scorer
  charts/latency_distribution.png   — per-case latency score distribution
  charts/failures_by_category.png   — count of sub-threshold cases per category

Usage:
    python agent-evaluation/visualize_metrics.py
    python agent-evaluation/visualize_metrics.py --metrics eval_metrics.json --threshold 1.0
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

DEFAULT_METRICS_PATH = Path(__file__).resolve().parent / "eval_metrics.json"
CHARTS_DIR = Path(__file__).resolve().parent / "charts"
DEFAULT_THRESHOLD = 1.0

PALETTE = {
    "pass": "#4CAF50",
    "warn": "#FF9800",
    "fail": "#F44336",
    "bar":  "#5C6BC0",
    "bg":   "#FAFAFA",
}

_SCORER_LABELS = {
    "BenchmarkSimilarityValidity": "Benchmark Similarity Validity",
    "ClassificationValidity":      "Classification Validity",
    "ClauseStructureValidity":     "Clause Structure Validity",
    "ExpectedClauseType":          "Expected Clause Type",
    "Factuality":                  "Factuality",
    "Latency":                     "Latency",
    "NoError":                     "No Error",
    "OutputStructureValidity":     "Output Structure Validity",
    "RiskFactorsPresence":         "Risk Factors Presence",
    "RiskScoreValidity":           "Risk Score Validity",
}

_CATEGORY_LABELS = {
    "agency_agreement":          "Agency Agreement",
    "commercial_lease":          "Commercial Lease",
    "consulting_agreement":      "Consulting Agreement",
    "data_processing_agreement": "Data Processing Agreement",
    "distribution_agreement":    "Distribution Agreement",
    "distributor_agreement":     "Distributor Agreement",
    "employment_contract":       "Employment Contract",
    "enterprise_software":       "Enterprise Software",
    "franchise_agreement":       "Franchise Agreement",
    "joint_venture":             "Joint Venture",
    "master_service_agreement":  "Master Service Agreement",
    "nda":                       "NDA",
    "nda_services_agreement":    "NDA Services Agreement",
    "partnership_agreement":     "Partnership Agreement",
    "research_collaboration":    "Research Collaboration",
    "reseller_agreement":        "Reseller Agreement",
    "saas_agreement":            "SaaS Agreement",
    "services_agreement":        "Services Agreement",
    "software_license":          "Software License",
    "strategic_alliance":        "Strategic Alliance",
    "supply_agreement":          "Supply Agreement",
    "technology_licensing":      "Technology Licensing",
    "technology_transfer":       "Technology Transfer",
}


def _fmt(name: str, mapping: dict) -> str:
    return mapping.get(name, name.replace("_", " ").title())


def load_metrics(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(p) as f:
        return json.load(f)


def _save(fig: plt.Figure, name: str) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out = CHARTS_DIR / name
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Chart 1: Overall scores bar chart
# ---------------------------------------------------------------------------

def plot_overall_scores(metrics: dict) -> None:
    overall = metrics.get("overall_scores", {})
    scorers = sorted(overall.keys())
    averages = [overall[s]["average"] for s in scorers]
    counts = [overall[s]["count"] for s in scorers]

    colors = [
        PALETTE["pass"] if v == 1.0
        else PALETTE["warn"] if v >= 0.75
        else PALETTE["fail"]
        for v in averages
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    scorer_labels = [_fmt(s, _SCORER_LABELS) for s in scorers]
    bars = ax.barh(scorer_labels, averages, color=colors, edgecolor="white", height=0.6)

    for bar, avg, n in zip(bars, averages, counts):
        ax.text(
            min(avg + 0.005, 1.01), bar.get_y() + bar.get_height() / 2,
            f"{avg:.3f}  (n={n})",
            va="center", ha="left", fontsize=9, color="#333333",
        )

    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Average Score", fontsize=11)
    ax.set_title("Overall Scorer Averages", fontsize=13, fontweight="bold", pad=12)
    ax.axvline(1.0, color="#BDBDBD", linewidth=0.8, linestyle="--")
    ax.tick_params(axis="y", labelsize=10)
    ax.spines[["top", "right", "bottom"]].set_visible(False)

    total = metrics.get("total_cases", 0)
    ax.text(
        0.99, -0.08, f"n = {total} total cases",
        transform=ax.transAxes, ha="right", fontsize=9, color="#757575",
    )

    _save(fig, "overall_scores.png")


# ---------------------------------------------------------------------------
# Chart 2: Category × Scorer heatmap
# ---------------------------------------------------------------------------

def plot_category_heatmap(metrics: dict) -> None:
    per_category = metrics.get("per_category", {})
    categories = sorted(per_category.keys())

    all_scorers: set[str] = set()
    for cat_data in per_category.values():
        all_scorers.update(cat_data.keys())
    scorers = sorted(all_scorers)

    matrix = np.full((len(categories), len(scorers)), np.nan)
    for r, cat in enumerate(categories):
        for c, scorer in enumerate(scorers):
            entry = per_category[cat].get(scorer)
            if entry is not None:
                matrix[r, c] = entry["average"]

    fig, ax = plt.subplots(figsize=(max(10, len(scorers) * 1.3), max(6, len(categories) * 0.55)))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "eval", [PALETTE["fail"], PALETTE["warn"], PALETTE["pass"]]
    )
    cmap.set_bad(color="#E0E0E0")

    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(scorers)))
    ax.set_xticklabels([_fmt(s, _SCORER_LABELS) for s in scorers], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([_fmt(c, _CATEGORY_LABELS) for c in categories], fontsize=9)

    for r in range(len(categories)):
        for c in range(len(scorers)):
            val = matrix[r, c]
            if not np.isnan(val):
                text_color = "white" if val < 0.6 else "#222222"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color=text_color)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Average Score")
    ax.set_title("Score Heatmap: Category × Scorer", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(length=0)

    _save(fig, "category_heatmap.png")


# ---------------------------------------------------------------------------
# Chart 3: Per-case Latency score distribution
# ---------------------------------------------------------------------------

def plot_latency_distribution(metrics: dict) -> None:
    latency_scores = [
        case["scores"].get("Latency")
        for case in metrics.get("per_case", [])
        if case.get("scores", {}).get("Latency") is not None
    ]

    if not latency_scores:
        print("  no latency scores found — skipping latency chart")
        return

    buckets = {1.0: 0, 0.75: 0, 0.5: 0, 0.25: 0, 0.0: 0}
    for s in latency_scores:
        buckets[s] = buckets.get(s, 0) + 1

    labels = {
        1.0: "< 30s\n(1.0)",
        0.75: "30–60s\n(0.75)",
        0.5:  "60–120s\n(0.5)",
        0.25: "120–180s\n(0.25)",
        0.0:  "> 180s\n(0.0)",
    }
    ordered = [1.0, 0.75, 0.5, 0.25, 0.0]
    counts = [buckets.get(v, 0) for v in ordered]
    tick_labels = [labels[v] for v in ordered]
    colors = [PALETTE["pass"], PALETTE["warn"], PALETTE["warn"], PALETTE["fail"], PALETTE["fail"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bars = ax.bar(tick_labels, counts, color=colors, edgecolor="white", width=0.55)
    for bar, count in zip(bars, counts):
        if count:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(count), ha="center", va="bottom", fontsize=10,
            )

    ax.set_ylabel("Number of Cases", fontsize=11)
    ax.set_title("Latency Score Distribution (per case)", fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])

    _save(fig, "latency_distribution.png")


# ---------------------------------------------------------------------------
# Chart 4: Failures per category (cases below threshold)
# ---------------------------------------------------------------------------

def plot_failures_by_category(metrics: dict, threshold: float) -> None:
    failure_counts: dict[str, int] = {}

    for case in metrics.get("per_case", []):
        cat = case.get("category", "unknown")
        has_failure = any(
            s < threshold for s in case.get("scores", {}).values()
        ) or bool(case.get("error"))
        if has_failure:
            failure_counts[cat] = failure_counts.get(cat, 0) + 1

    if not failure_counts:
        print(f"  no failures at threshold {threshold} — skipping failures chart")
        return

    categories = sorted(failure_counts, key=lambda c: -failure_counts[c])
    counts = [failure_counts[c] for c in categories]
    category_labels = [_fmt(c, _CATEGORY_LABELS) for c in categories]

    fig, ax = plt.subplots(figsize=(9, max(4, len(categories) * 0.5)))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    ax.barh(category_labels, counts, color=PALETTE["fail"], edgecolor="white", height=0.6)
    for i, (cat, count) in enumerate(zip(category_labels, counts)):
        ax.text(count + 0.05, i, str(count), va="center", fontsize=10)

    ax.set_xlabel("Number of Failed Cases", fontsize=11)
    ax.set_title(
        f"Failures by Category  (threshold < {threshold})",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.set_xlim(0, max(counts) + 1.5)
    ax.tick_params(axis="y", labelsize=9)

    _save(fig, "failures_by_category.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize eval_metrics.json. Charts saved to charts/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--metrics", type=str, default=str(DEFAULT_METRICS_PATH),
        help=f"Path to eval_metrics.json (default: {DEFAULT_METRICS_PATH})",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Failure threshold for the failures-by-category chart (default: {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    metrics = load_metrics(args.metrics)
    print(f"\nGenerating charts from {args.metrics} → {CHARTS_DIR}/\n")

    plot_overall_scores(metrics)
    plot_category_heatmap(metrics)
    plot_latency_distribution(metrics)
    plot_failures_by_category(metrics, args.threshold)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
