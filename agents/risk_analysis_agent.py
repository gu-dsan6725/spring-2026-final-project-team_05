"""Risk Analysis Agent: scores clause risk using CUAD taxonomy as a structured rubric."""

import json
import os
import re
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from state import ContractState
from observability import get_logger 

TAXONOMY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cuad", "taxonomy.json")

CUAD_RUBRIC = {}
if os.path.exists(TAXONOMY_PATH):
    with open(TAXONOMY_PATH) as f:
        for entry in json.load(f):
            CUAD_RUBRIC[entry["name"]] = entry["question"]

SYSTEM_PROMPT = """You are a legal risk analyst for commercial contracts.

Given a contract clause, its classified type, and a legal review rubric for that clause type,
evaluate the risk level by considering:
1. Ambiguous or vague language that could lead to unfavorable interpretations
2. Missing protective provisions that are standard for this clause type
3. Deviation from standard phrasing or industry norms
4. The specific legal review criteria provided in the rubric

{rubric_section}

Respond with ONLY valid JSON in this exact format:
{{
    "risk_score": 0.0 to 1.0 (0 = low risk, 1 = high risk),
    "risk_factors": ["factor 1", "factor 2"],
    "reasoning": "brief explanation of the overall risk assessment"
}}"""

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Clause type: {clause_type}\n\nClause text:\n{clause_text}"),
])

chain = prompt | llm


def analyze_risk(clause_text: str, clause_type: str) -> dict:
    """Analyze risk for a single clause using CUAD rubric."""
    rubric_question = CUAD_RUBRIC.get(clause_type, "")
    if rubric_question:
        rubric_section = f"Legal review rubric for this clause type:\n{rubric_question}"
    else:
        rubric_section = "No specific rubric available for this clause type. Use general legal risk analysis."

    response = chain.invoke({
        "rubric_section": rubric_section,
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
            "risk_score": 0.0,
            "risk_factors": [],
            "reasoning": "Failed to parse LLM response",
        }

    return result


def risk_analysis_node(state: ContractState) -> dict:
    """LangGraph node: analyze risk for all classified clauses."""
    risk_scored = []

    for clause in state["classified_clauses"]:
        result = analyze_risk(clause["text"], clause.get("clause_type", "Other"))
        risk_scored.append({
            **clause,
            "risk_score": result.get("risk_score", 0.0),
            "risk_factors": result.get("risk_factors", []),
        })

    # observability: log risk analysis summary as Braintrust span
    logger = get_logger()
    if logger:
        with logger.start_span("risk_analysis_node") as span:
            risk_scores = [c["risk_score"] for c in risk_scored]
            span.log(
                input={"clauses_received": len(state["classified_clauses"])},
                output={
                    "clauses_scored": len(risk_scored),
                    "high_risk_count": sum(1 for s in risk_scores if s >= 0.7),
                    "avg_risk_score": sum(risk_scores) / len(risk_scores) if risk_scores else 0.0,
                },
                metadata={
                    "risk_scores": risk_scores,
                },
            )

    return {"risk_scores": risk_scored}
