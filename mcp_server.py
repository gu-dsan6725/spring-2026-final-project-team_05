"""
MCP Server for the contract analysis pipeline - this exposes the full LangGraph 
pipeline as FastMCP tools so any MCP-compatible clients (Claude Desktop, Cursor, etc.) 
can analyze contracts directly

Tools:
  - analyze_contract: run the pipeline on a raw contract string
  - analyze_contract_file: run the pipeline on a local .txt file path

Usage: 
    stdio transport, default for MCP clients: `python mcp_server.py`
    HTTP transport, for testing in a browser/curl: `python mcp_server.py --transport http --port 8000`

Prereqs:
  - ANTHROPIC_API_KEY in .env or environment
  - CUAD vector store built: python scripts/build_vector_store.py
"""

import json
import os

from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

# initialize FastMCP server with name
mcp = FastMCP(
    name="contract-analysis",
    instructions=(
        "Analyzes commercial contracts using a multi-agent AI pipeline. "
        "Classifies clauses by CUAD type, scores risk, and benchmarks against "
        "real contract examples. Returns a structured JSON report."
    ),
)

@mcp.tool()
def analyze_contract(contract_text: str) -> str:
    """
    Run full contract analysis pipeline on a raw contract string.
    All pipeline stages: ingestion -> knowledge graph -> classification -> risk analysis 
    -> benchmark -> report

    Args: contract_text - full text of contract to analyze

    Returns:
        A JSON string containing the analysis report
    """
    from agents.orchestrator_agent import run_pipeline

    result = run_pipeline(contract_text)
    return result.get("report", json.dumps({"error": "Pipeline returned no report"}))

@mcp.tool()
def analyze_contract_file(file_path: str) -> str:
    """
    Run full contract analysis pipeline on a local .txt file
    Reads file at given path & passes contents through the analysis pipeline (could
    be useful when contract is stored on disk)

    Args: file_path - absolute or relative path to a .txt contract file.

    Returns:
        A JSON string containing the analysis report (same format as analyze_contract)
        Returns error JSON if file not found
    """
    if not os.path.exists(file_path):
        return json.dumps({"error": f"File not found: {file_path}"})

    with open(file_path, encoding="utf-8") as f:
        contract_text = f.read()

    from agents.orchestrator_agent import run_pipeline

    result = run_pipeline(contract_text)
    return result.get("report", json.dumps({"error": "Pipeline returned no report"}))

@mcp.resource("contract-analysis://pipeline-info")
def pipeline_info() -> str:
    """
    Describe the contract analysis pipeline and its output schema.
    Returns plain-text summary of what the pipeline does, what fields
    each clause in the report contains
    """
    return (
        "Contract Analysis Pipeline\n"
        "==========================\n"
        "Stages: ingestion -> knowledge_graph -> classification -> risk_analysis -> benchmark -> report\n\n"
        "Output report fields per clause:\n"
        "  - id                  : clause index\n"
        "  - section             : section header text\n"
        "  - clause_type         : CUAD taxonomy label (41 types)\n"
        "  - confidence          : classification confidence (0.0–1.0)\n"
        "  - risk_score          : risk level (0.0 = low, 1.0 = high)\n"
        "  - risk_factors        : list of identified risk factors\n"
        "  - benchmark_similarity: similarity to CUAD norms (0.0–1.0)\n"
        "  - benchmark_source    : source CUAD contract(s) used\n"
        "  - text                : first 200 chars of clause text\n\n"
        "Summary fields:\n"
        "  - total_clauses       : number of clauses analyzed\n"
        "  - entities_extracted  : knowledge graph entity count\n"
        "  - relationships_extracted: knowledge graph relationship count\n"
        "  - graph_image_path    : path to saved knowledge graph PNG\n"
    )

if __name__ == "__main__":
    mcp.run()
