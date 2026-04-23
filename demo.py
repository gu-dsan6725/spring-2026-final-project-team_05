"""
Demo- run full contract analysis pipeline end-to-end

Usage:
    python demo.py # uses default sample contract
    python demo.py path/to/contract.txt  # use a specific contract

Prereqs:
    1. Set ANTHROPIC_API_KEY in .env or environment
    2. Build CUAD vector store once: python scripts/build_vector_store.py

Output:
    - Prints summary report to the terminal
    - Saves full JSON report to pipeline_output.json
"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from agents.orchestrator_agent import run_pipeline

DEFAULT_CONTRACT = os.path.join(
    "data", "contracts",
    "AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agreement.txt"
)


def print_report(report: dict, contract_name: str) -> None:
    clauses = report["clauses"]
    total = report["summary"]["total_clauses"]
    high_risk = [c for c in clauses if c.get("risk_score", 0) >= 0.7]

    print("=" * 60)
    print("CONTRACT ANALYSIS REPORT")
    print("=" * 60)
    print(f"Contract: {contract_name}")
    print(f"Total clauses analyzed: {total}")
    print(f"High-risk clauses (score >= 0.7): {len(high_risk)}")
    print()

    sorted_clauses = sorted(clauses, key=lambda c: c.get("risk_score", 0), reverse=True)

    print("--- TOP 5 CLAUSES BY RISK SCORE ---")
    for clause in sorted_clauses[:5]:
        risk = clause.get("risk_score", 0)
        sim = clause.get("benchmark_similarity", 0)
        ctype = clause.get("clause_type", "Unknown")
        section = clause.get("section", "")[:60]
        source = clause.get("benchmark_source", "")
        factors = clause.get("risk_factors", [])
        text_preview = clause.get("text", "")[:180].strip().replace("\n", " ")

        print(f"\n  [{ctype}]")
        print(f"  Section:   {section}")
        print(f"  Risk:      {risk:.2f}  |  Benchmark similarity: {sim:.2f}")
        print(f"  Source:    {source}")
        if factors:
            print(f"  Factors:   {'; '.join(factors[:2])}")
        print(f"  Text:      {text_preview}...")

    print()

def main():
    contract_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONTRACT

    if not os.path.exists(contract_path):
        print(f"Error: contract file not found: {contract_path}")
        sys.exit(1)

    contract_name = os.path.basename(contract_path)
    print(f"Analyzing: {contract_name}")
    print("Running pipeline: ingestion -> classification -> risk analysis -> benchmark -> report")
    print("(This may take a minute...)\n")

    with open(contract_path, encoding="utf-8") as f:
        contract_text = f.read()

    result = run_pipeline(contract_text)
    report = json.loads(result["report"])

    print_report(report, contract_name)

    output_path = "pipeline_output.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full JSON report saved to: {output_path}")

if __name__ == "__main__":
    main()
