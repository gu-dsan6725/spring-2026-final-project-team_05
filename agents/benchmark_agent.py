"""Benchmark Agent: compares clauses against real CUAD contract examples.

RECENT UPDATE: Hybrid Retrieval (BM25 + ChromaDB):
Now uses 2 retrieval methods & merges  results before passing to the LLM:
  - ChromaDB (semantic/vector search): finds conceptually similar clauses
  - BM25 (keyword search): finds clauses with exact or near-exact matching terms
"""

import json
import os
import re

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
from state import ContractState

STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cuad_vector_store")
COLLECTION_NAME = "cuad_contracts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
N_RESULTS = 3  # examples to retrieve per method (semantic + keyword)

SYSTEM_PROMPT = """You are a legal benchmarking analyst for commercial contracts.

Given a contract clause, its classified type, and examples of similar clauses retrieved
from real CUAD commercial contracts, evaluate how standard the clause is.

The examples below come from two retrieval methods:
- Semantic matches: conceptually similar clauses found via vector search
- Keyword matches: clauses with similar exact phrasing found via keyword search

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

# BM25 index loaded once at module import
_bm25_index = None
_bm25_corpus = None  # list of {"text": ..., "source": ...}


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


# Load BM25 index from chunks.json saved by build_vector_store.py
def _get_bm25():
    global _bm25_index, _bm25_corpus
    if _bm25_index is None:
        chunks_path = os.path.join(STORE_DIR, "chunks.json")
        try:
            with open(chunks_path, encoding="utf-8") as f:
                _bm25_corpus = json.load(f)
            tokenized = [doc["text"].lower().split() for doc in _bm25_corpus]
            _bm25_index = BM25Okapi(tokenized)
        except Exception as e:
            print(f"[benchmark_agent] BM25 index not available: {e}")
    return _bm25_index, _bm25_corpus

def _retrieve_semantic(clause_text: str, clause_type: str) -> list[dict]:
    """Retrieve top-N semantically similar CUAD examples via ChromaDB"""
    collection = _get_collection()
    if collection is None:
        return []
    query = f"{clause_type}: {clause_text[:500]}"
    results = collection.query(query_texts=[query], n_results=N_RESULTS)
    return [
        {"text": doc, "source": meta["source"]}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


# Keyword retrieval w/ BM25
def _retrieve_bm25(clause_text: str, clause_type: str) -> list[dict]:
    """Retrieve top-N keyword-matched CUAD examples via BM25"""
    bm25, corpus = _get_bm25()
    if bm25 is None:
        return []
    query_tokens = f"{clause_type} {clause_text[:500]}".lower().split()
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N_RESULTS]
    return [{"text": corpus[i]["text"], "source": corpus[i]["source"]} for i in top_indices]

# Merge semantic + BM25 results, dedupe, format for LLM prompt
def _retrieve_cuad_examples(clause_text: str, clause_type: str) -> tuple[str, list[str]]:
    """Hybrid retrieval: combine ChromaDB (semantic) and BM25 (keyword) results"""
    semantic_results = _retrieve_semantic(clause_text, clause_type)
    bm25_results = _retrieve_bm25(clause_text, clause_type)

    # Deduplicate by first 80 chars of text (same chunk shouldn't appear twice)
    seen = set()
    unique_semantic, unique_bm25 = [], []
    for item in semantic_results:
        key = item["text"][:80]
        if key not in seen:
            seen.add(key)
            unique_semantic.append(item)
    for item in bm25_results:
        key = item["text"][:80]
        if key not in seen:
            seen.add(key)
            unique_bm25.append(item)

    sources = [r["source"] for r in unique_semantic + unique_bm25]

    # Format examples for the LLM prompt, labelled by retrieval method
    parts = []
    for i, r in enumerate(unique_semantic):
        parts.append(f"Semantic match {i + 1} (source: {r['source']}):\n{r['text'][:400]}")
    for i, r in enumerate(unique_bm25):
        parts.append(f"Keyword match {i + 1} (source: {r['source']}):\n{r['text'][:400]}")

    examples_text = "\n\n".join(parts)
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
