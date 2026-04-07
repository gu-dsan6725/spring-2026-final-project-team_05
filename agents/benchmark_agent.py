"""Benchmark Agent: compares clauses against industry standard contract language."""

import json
import re
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from state import ContractState

SYSTEM_PROMPT = """You are a legal benchmarking analyst for commercial contracts.

Given a contract clause and its classified type, compare it against standard industry
language for that clause type. Evaluate how the clause compares to typical provisions
found in commercial contracts (NDAs, MSAs, licensing agreements, vendor agreements, etc.).

Respond with ONLY valid JSON in this exact format:
{{
    "benchmark_similarity": 0.0 to 1.0 (0 = highly unusual, 1 = very standard),
    "deviations": ["deviation 1", "deviation 2"],
    "standard_language_summary": "brief description of what is typical for this clause type",
    "reasoning": "brief explanation of how this clause compares to industry norms"
}}"""

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Clause type: {clause_type}\n\nClause text:\n{clause_text}"),
])

chain = prompt | llm


def benchmark_clause(clause_text: str, clause_type: str) -> dict:
    """Benchmark a single clause against industry norms."""
    response = chain.invoke({
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

    return result


def benchmark_node(state: ContractState) -> dict:
    """LangGraph node: benchmark all risk-scored clauses."""
    benchmarked = []

    source = state.get("risk_scores", state.get("classified_clauses", []))

    for clause in source:
        result = benchmark_clause(clause["text"], clause.get("clause_type", "Other"))
        benchmarked.append({
            **clause,
            "benchmark_similarity": result.get("benchmark_similarity", 0.0),
            "benchmark_source": "CUAD industry norms",
        })

    return {"benchmark_results": benchmarked}
