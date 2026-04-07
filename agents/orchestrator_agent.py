"""Orchestrator Agent: coordinates the contract analysis pipeline via LangGraph."""

import json
from langgraph.graph import StateGraph, END
from state import ContractState
from agents.ingestion_agent import ingestion_node
from agents.classification_agent import classification_node
from agents.risk_analysis_agent import risk_analysis_node
from agents.benchmark_agent import benchmark_node


def report_node(state: ContractState) -> dict:
    """Compile all agent outputs into a structured risk report."""
    clauses = state.get("benchmark_results", state.get("risk_scores", []))

    high_risk = [c for c in clauses if c.get("risk_score", 0) >= 0.7]
    medium_risk = [c for c in clauses if 0.4 <= c.get("risk_score", 0) < 0.7]
    low_risk = [c for c in clauses if c.get("risk_score", 0) < 0.4]

    report = {
        "summary": {
            "total_clauses": len(clauses),
            "high_risk": len(high_risk),
            "medium_risk": len(medium_risk),
            "low_risk": len(low_risk),
        },
        "clauses": [
            {
                "id": c.get("id"),
                "section": c.get("section", ""),
                "clause_type": c.get("clause_type", ""),
                "confidence": c.get("confidence", 0.0),
                "risk_score": c.get("risk_score", 0.0),
                "risk_factors": c.get("risk_factors", []),
                "benchmark_similarity": c.get("benchmark_similarity", 0.0),
                "benchmark_source": c.get("benchmark_source", ""),
                "text": c.get("text", "")[:200],
            }
            for c in clauses
        ],
    }

    return {"report": json.dumps(report, indent=2)}


def build_pipeline() -> StateGraph:
    """Build and compile the full LangGraph pipeline."""
    graph = StateGraph(ContractState)

    graph.add_node("ingestion", ingestion_node)
    graph.add_node("classification", classification_node)
    graph.add_node("risk_analysis", risk_analysis_node)
    graph.add_node("benchmark", benchmark_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion", "classification")
    graph.add_edge("classification", "risk_analysis")
    graph.add_edge("risk_analysis", "benchmark")
    graph.add_edge("benchmark", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_pipeline(contract_text: str) -> dict:
    pipeline = build_pipeline()
    result = pipeline.invoke({"raw_text": contract_text})
    return result
