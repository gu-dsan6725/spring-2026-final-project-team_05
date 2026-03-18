"""
pipeline.py

LangGraph pipeline, connects ingestion -> classification for now
Usage: python pipeline.py path/to/contract.txt
"""

import json
import sys
from langgraph.graph import StateGraph, END
from state import ContractState
from agents.ingestion_agent import ingestion_node
from agents.classification_agent import classification_node

def build_graph() -> StateGraph:
    graph = StateGraph(ContractState)
    graph.add_node("ingestion", ingestion_node)
    graph.add_node("classification", classification_node)
    graph.add_edge("ingestion", "classification")
    graph.add_edge("classification", END)
    graph.set_entry_point("ingestion")
    return graph.compile()

def run(contract_path: str) -> ContractState:
    raw_text = open(contract_path, encoding="utf-8").read()
    app = build_graph()
    result = app.invoke({"raw_text": raw_text})
    return result

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/contracts/AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agreement.txt"
    result = run(path)
    print(f"Clauses found:      {len(result['clauses'])}")
    print(f"Clauses classified: {len(result['classified_clauses'])}")
    print()

    for clause in result["classified_clauses"][:5]:
        print(f"[{clause['clause_type']}] (conf: {clause.get('confidence', '?')})")
        print(f"  {clause['text'][:250].strip()}")
        print()

    # full output
    # with open("pipeline_output.json", "w") as f:
    #     json.dump(result["classified_clauses"], f, indent=2)