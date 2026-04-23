"""Orchestrator Agent: coordinates the contract analysis pipeline via LangGraph."""

import json
from langgraph.graph import StateGraph, END
from state import ContractState
from agents.ingestion_agent import ingestion_node
from agents.classification_agent import classification_node
from agents.risk_analysis_agent import risk_analysis_node
from agents.benchmark_agent import benchmark_node
from agents.knowledge_graph_agent import knowledge_graph_node
from observability import get_logger


def report_node(state: ContractState) -> dict:
    """Compile all agent outputs into a structured risk report."""
    clauses = state.get("benchmark_results", state.get("risk_scores", []))

    report = {
        "summary": {
            "total_clauses": len(clauses),
            "entities_extracted": len(state.get("entities", [])),
            "relationships_extracted": len(state.get("relationships", [])),
            "graph_image_path": state.get("graph_image_path", ""),
        },
        # entities and relationships in report
        "entities": state.get("entities", []),
        "relationships": state.get("relationships", []),
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
    graph.add_node("knowledge_graph", knowledge_graph_node)
    graph.add_node("classification", classification_node)
    graph.add_node("risk_analysis", risk_analysis_node)
    graph.add_node("benchmark", benchmark_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion", "knowledge_graph")
    graph.add_edge("knowledge_graph", "classification")
    graph.add_edge("classification", "risk_analysis")
    graph.add_edge("risk_analysis", "benchmark")
    graph.add_edge("benchmark", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_pipeline(contract_text: str) -> dict:
    pipeline = build_pipeline()

    # updated for observability -- wrap whole pipeline run in top-level baintrust span
    logger = get_logger()
    if logger:
        with logger.start_span("contract_pipeline") as span:
            span.log(
                input={"contract_length_chars": len(contract_text)},
                metadata={"pipeline_nodes": [
                    "ingestion", "knowledge_graph", "classification",
                    "risk_analysis", "benchmark", "report"
                ]},
            )
            result = pipeline.invoke({"raw_text": contract_text})
            report_data = json.loads(result.get("report", "{}"))
            span.log(
                output={
                    "total_clauses": report_data.get("summary", {}).get("total_clauses", 0),
                    "entities_extracted": report_data.get("summary", {}).get("entities_extracted", 0),
                }
            )
    else:
        result = pipeline.invoke({"raw_text": contract_text})

    return result
