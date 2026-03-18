from typing import TypedDict, List


class Clause(TypedDict, total=False):
    id: int
    text: str
    section: str
    clause_type: str
    confidence: float
    risk_score: float
    risk_factors: List[str]
    benchmark_similarity: float
    benchmark_source: str


class ContractState(TypedDict, total=False):
    raw_text: str
    clauses: List[Clause]
    classified_clauses: List[Clause]
    risk_scores: List[Clause]
    benchmark_results: List[Clause]
    report: str
