"""Classification Agent: tags contract clauses using the CUAD taxonomy."""

import json
import re
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from state import ContractState
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

with open("data/cuad/taxonomy.json") as f:
    TAXONOMY = json.load(f)

CUAD_CLAUSE_TYPES = [entry["name"] for entry in TAXONOMY]

SYSTEM_PROMPT = """You are a legal clause classifier for commercial contracts.

Given a contract clause, classify it into one or more of the following CUAD clause types:

{clause_types}

If the clause does not match any type, classify it as "Other".

Respond with ONLY valid JSON in this exact format:
{{
    "clause_type": "the primary clause type",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}"""

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=256)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Classify this clause:\n\n{clause_text}"),
])

chain = prompt | llm


def classify_clause(clause_text: str) -> dict:
    """Classify a single clause and return structured result."""
    response = chain.invoke({
        # "clause_types": "\n".join(f"- {ct}" for ct in CUAD_CLAUSE_TYPES),
        "clause_types": "\n".join(
            f"- {entry['name']}: {entry['question'].split('Details: ')[-1]}"
            for entry in TAXONOMY
        ),
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
            "clause_type": "Other",
            "confidence": 0.0,
            "reasoning": "Failed to parse LLM response",
        }

    return result


def classification_node(state: ContractState) -> dict:
    """LangGraph node: classify all clauses from the ingestion agent."""
    classified = []

    for clause in state["clauses"][:3]: # changed from state["clauses"] for now to conserve API credits
        result = classify_clause(clause["text"])
        classified.append({
            **clause,
            "clause_type": result["clause_type"],
            "confidence": result.get("confidence", 0.0),
        })

    return {"classified_clauses": classified}
