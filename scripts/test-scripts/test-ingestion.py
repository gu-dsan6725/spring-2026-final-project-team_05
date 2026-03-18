import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from agents.ingestion_agent import split_into_clauses

text = open("data/contracts/ACCURAYINC_09_01_2010-EX-10.31-DISTRIBUTOR AGREEMENT.txt").read()

clauses = split_into_clauses(text)
print(len(clauses))

for c in clauses[:3]:
    print(c["section"], "—", c["text"][:80])