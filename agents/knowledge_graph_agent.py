"""Knowledge Graph Agent: extracts entities and relationships from contract text.

Uses an LLM to gather main entities (parties, dates, amounts, products etc) and their
relationships from the contract opening section, builds & saves a NetworkX graph
visualization. 

Inserted into the pipeline between ingestion and classification.
Output stored in state as: entities, relationships, graph_image_path.
"""

import json
import os
import re

import matplotlib
matplotlib.use("Agg") # non-interactive backend (no display)
import matplotlib.pyplot as plt
import networkx as nx
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from state import ContractState

OUTPUT_DIR = "outputs"

# Only use first ~5000 chars of the contract for info on parties/dates/obligations
TEXT_EXCERPT_CHARS = 5000

SYSTEM_PROMPT = """You are a legal entity extractor for commercial contracts.

Given the opening section of a contract, extract the most important entities and
relationships. Focus on who the parties are, what they are agreeing to do, and
any key constraints (dates, amounts, exclusivity, territory, etc.).

Respond with ONLY valid JSON in this exact format:
{{
    "entities": [
        {{"name": "short name", "type": "party|date|amount|product|location|other"}}
    ],
    "relationships": [
        {{"source": "entity name", "relation": "short verb phrase", "target": "entity name"}}
    ]
}}

Aim for 5-10 entities and 5-10 relationships. Keep entity names short (2-4 words max).
Only include relationships between entities you listed."""

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1024)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Contract opening section:\n\n{contract_text}"),
])

chain = prompt | llm

# color scheme for entity types in graph viz
ENTITY_COLORS = {
    "party": "#4C72B0",
    "date": "#DD8452",
    "amount": "#55A868",
    "product": "#8172B2",
    "location": "#937860",
    "other": "#C44E52",
}

def extract_entities_and_relationships(contract_text: str) -> tuple[list, list]:
    """Call LLM to extract entities & relationships from contract opening"""
    excerpt = contract_text[:TEXT_EXCERPT_CHARS].strip()

    response = chain.invoke({"contract_text": excerpt})

    try:
        text = response.content.strip()
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {"entities": [], "relationships": []}

    return result.get("entities", []), result.get("relationships", [])

def build_and_save_graph(entities: list, relationships: list) -> str:
    """Build directed NetworkX graph, save PNG"""
    if not entities:
        return ""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "knowledge_graph.png")

    G = nx.DiGraph()

    for entity in entities:
        G.add_node(entity["name"], entity_type=entity.get("type", "other"))

    for rel in relationships:
        src, tgt = rel.get("source", ""), rel.get("target", "")
        if src and tgt and src in G.nodes and tgt in G.nodes:
            G.add_edge(src, tgt, label=rel.get("relation", ""))

    if len(G.nodes) == 0:
        return ""

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=2.5)

    node_colors = [
        ENTITY_COLORS.get(G.nodes[n].get("entity_type", "other"), ENTITY_COLORS["other"])
        for n in G.nodes
    ]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, edge_color="#888888", width=1.5)
    edge_labels = {(u, v): d["label"][:25] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, font_color="#333333")

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=10, label=etype)
        for etype, color in ENTITY_COLORS.items()
    ]
    plt.legend(handles=legend_handles, loc="upper left", fontsize=8)

    plt.title("Contract Knowledge Graph", fontsize=14, pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path

def knowledge_graph_node(state: ContractState) -> dict:
    """LangGraph node- extract entities/relationships & build knowledge graph"""
    entities, relationships = extract_entities_and_relationships(state["raw_text"])
    graph_path = build_and_save_graph(entities, relationships)

    return {
        "entities": entities,
        "relationships": relationships,
        "graph_image_path": graph_path,
    }
