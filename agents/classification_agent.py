"""Classification Agent: tags contract clauses using the CUAD taxonomy."""

import json
import re
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from state import ContractState

CUAD_CLAUSE_TYPES = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
    "Indemnification",
]

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
        "clause_types": "\n".join(f"- {ct}" for ct in CUAD_CLAUSE_TYPES),
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

    for clause in state["clauses"]:
        result = classify_clause(clause["text"])
        classified.append({
            **clause,
            "clause_type": result["clause_type"],
            "confidence": result.get("confidence", 0.0),
        })

    return {"classified_clauses": classified}
