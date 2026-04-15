"""Benchmark Agent: compares clauses against real CUAD contract examples."""

import json
import os
import re

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from state import ContractState

STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cuad_vector_store")
COLLECTION_NAME = "cuad_contracts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
N_RESULTS = 3  # number of CUAD examples to retrieve per clause

SYSTEM_PROMPT = """You are a legal benchmarking analyst for commercial contracts.

Given a contract clause, its classified type, and examples of similar clauses retrieved
from real CUAD commercial contracts, evaluate how standard the clause is.

{examples_section}

Respond with ONLY valid JSON in this exact format:
{{
    "benchmark_similarity": 0.0 to 1.0 (0 = highly unusual, 1 = very standard),
    "deviations": ["deviation 1", "deviation 2"],
    "standard_language_summary": "brief description of what is typical for this clause type",
    "reasoning": "brief explanation of how this clause compares to the CUAD examples"
}}"""

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Clause type: {clause_type}\n\nClause text:\n{clause_text}"),
])

chain = prompt | llm

# Load vector store once at module import
_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        try:
            ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
            client = chromadb.PersistentClient(path=STORE_DIR)
            _collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)
        except Exception as e:
            print(f"[benchmark_agent] Warning: could not load CUAD vector store: {e}")
            print("[benchmark_agent] Run 'python scripts/build_vector_store.py' to build it.")
            print("[benchmark_agent] Falling back to LLM-only benchmarking.")
    return _collection


def _retrieve_cuad_examples(clause_text: str, clause_type: str) -> tuple[str, list[str]]:
    """Retrieve the most similar CUAD clause examples for context."""
    collection = _get_collection()
    if collection is None:
        return "", []

    query = f"{clause_type}: {clause_text[:500]}"
    results = collection.query(query_texts=[query], n_results=N_RESULTS)

    docs = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    examples_text = "\n\n".join(
        f"CUAD Example {i + 1} (source: {src}):\n{doc[:400]}"
        for i, (doc, src) in enumerate(zip(docs, sources))
    )
    return examples_text, sources


def benchmark_clause(clause_text: str, clause_type: str) -> dict:
    """Benchmark a single clause against CUAD examples."""
    examples_text, sources = _retrieve_cuad_examples(clause_text, clause_type)

    if examples_text:
        examples_section = f"Similar clauses retrieved from CUAD contracts:\n\n{examples_text}"
    else:
        examples_section = "No CUAD examples available. Use general legal knowledge."

    response = chain.invoke({
        "examples_section": examples_section,
        "clause_type": clause_type,
        "clause_text": clause_text,
    })

    try:
        text = response.content.strip()
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {
            "benchmark_similarity": 0.0,
            "deviations": [],
            "standard_language_summary": "",
            "reasoning": "Failed to parse LLM response",
        }

    result["_sources"] = sources
    return result


def benchmark_node(state: ContractState) -> dict:
    """LangGraph node: benchmark all risk-scored clauses against CUAD contracts."""
    benchmarked = []

    source = state.get("risk_scores", state.get("classified_clauses", []))

    for clause in source:
        result = benchmark_clause(clause["text"], clause.get("clause_type", "Other"))
        sources = result.get("_sources", [])
        benchmark_source = (
            "CUAD: " + ", ".join(s[:50] for s in sources[:2])
            if sources
            else "CUAD (LLM fallback. Run build_vector_store.py)"
        )
        benchmarked.append({
            **clause,
            "benchmark_similarity": result.get("benchmark_similarity", 0.0),
            "benchmark_source": benchmark_source,
        })

    return {"benchmark_results": benchmarked}
