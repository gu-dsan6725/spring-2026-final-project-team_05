"""Ingestion Agent: parses raw contract text into clause-level segments."""

import re
from state import ContractState, Clause

# Matches section headers (e.g. "1." "2.1" "SECTION 3" "ARTICLE IV")
SECTION_PATTERN = re.compile(
    r"(?:^|\n)(?:SECTION|ARTICLE|EXHIBIT)?\s*(?:\d+\.[\d\.]*|[IVXLC]+\.)\s+[A-Z]",
    re.IGNORECASE,
)

def split_into_clauses(text: str) -> list[Clause]:
    """Split contract text into clause segments on section headers."""
    matches = list(SECTION_PATTERN.finditer(text))

    if not matches:
        # Backup: split on double newlines (paragraph level) --> use for preamble
        segments = [s.strip() for s in re.split(r"\n{2,}", text) if s.strip()]
        return [
            Clause(id=i, text=seg, section=f"Paragraph {i + 1}")
            for i, seg in enumerate(segments)
        ]

    clauses: list[Clause] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[start:end].strip()
        if not segment:
            continue

        # Extract header (first line) as section label
        first_line = segment.splitlines()[0].strip()
        clauses.append(Clause(id=i, text=segment, section=first_line))

    return clauses


def ingestion_node(state: ContractState) -> dict:
    """LangGraph node: parse raw_text into clause segments."""
    clauses = split_into_clauses(state["raw_text"])
    return {"clauses": clauses}