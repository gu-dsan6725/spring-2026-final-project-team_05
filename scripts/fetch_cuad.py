"""
fetch_cuad.py

Downloads & saves the 3 CUAD data artifacts:
  1. ../data/cuad/taxonomy.json -> 41 clause categories + descriptions
  2. ../data/cuad/contracts/<title>.txt -> 510 full contract texts
  3. ../data/cuad/annotations.json -> all labeled spans, organized by contract

Dataset structure (SQuAD-format):
    Each row: {id, title, context, question, answers: {text: [...], answer_start: [...]}}
    510 unique contracts x 41 clause-category questions = ~20,950 rows total (train + test).
    The same contract appears 41 times -- once per question.
    Question format: "Highlight the parts (if any) of this contract related to
                      '{Category Name}' that should be reviewed by a lawyer."
"""

import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path("../data/cuad")
CONTRACTS_DIR = DATA_DIR / "../contracts"

# Official CUAD data release (SQuAD-format JSON) from The Atticus Project
_CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"

def load_cuad() -> list[dict]:
    """Download CUAD data.zip and return a flat list of SQuAD-format rows."""
    print("LOADING CUAD DATA FROM GITHUB")
    with urllib.request.urlopen(_CUAD_URL) as resp:
        data = resp.read()
    print(f"  downloaded {len(data) // 1024} KB")

    rows = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        json_files = [n for n in zf.namelist() if n.endswith(".json")]
        for name in json_files:
            squad = json.loads(zf.read(name))
            for article in squad["data"]:
                title = article["title"]
                for para in article["paragraphs"]:
                    context = para["context"]
                    for qa in para["qas"]:
                        rows.append({
                            "id": qa["id"],
                            "title": title,
                            "context": context,
                            "question": qa["question"],
                            "answers": {
                                "text": [a["text"] for a in qa["answers"]],
                                "answer_start": [a["answer_start"] for a in qa["answers"]],
                            },
                        })

    print(f"  {len(rows)} rows from {len(json_files)} file(s)")
    return rows

def extract_taxonomy(rows: list[dict]) -> None:
    """
    Extract all 41 clause categories

    Pattern for questions:
        "Highlight the parts (if any) of this contract related to
        '{Category Name}' that should be reviewed by a lawyer."

    The full question text is preserved as the category description since it is
    exactly the prompt the Classification Agent should use when labeling clauses

    Output:
        [{"id": 1, "name": "Document Name", "question": "<full question text>"}, ...]
    """
    seen: dict[str, str] = {} # name -> question text (deduped, insertion-ordered)

    for row in rows:
        question = row["question"]
        match = re.search(r'related to "(.+?)" that should be', question)
        if match:
            name = match.group(1)
            if name not in seen:
                seen[name] = question.strip()

    taxonomy = [
        {"id": i + 1, "name": name, "question": question}
        for i, (name, question) in enumerate(seen.items())
    ]

    out_path = DATA_DIR / "taxonomy.json"
    with open(out_path, "w") as f:
        json.dump(taxonomy, f, indent=2)

    print(f"  taxonomy.json: {len(taxonomy)} categories")


def extract_contracts(rows: list[dict]) -> None:
    """
    Extract the 510 unique contract full texts and save each as a .txt file.

    The `context` field holds the full contract text and is the same across
    all 41 rows for a given contract, so we just need the first occurrence.

    Output: data/cuad/contracts/<sanitized_title>.txt
    """
    CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()

    for row in rows:
        title = row["title"]
        if title in seen:
            continue
        seen.add(title)

        safe_name = re.sub(r'[<>:"/\\|?*]', "_", title)
        out_path = CONTRACTS_DIR / f"{safe_name}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(row["context"])

    print(f"  contracts/: {len(seen)} files")

def extract_annotations(rows: list[dict]) -> None:
    """
    Extract all the labeled spans and organize by contract title

    Each contract maps to a list of 41 annotation entries (one per category)
    An entry's `spans` list is empty if that clause type doesn't appear in the contract

    Output:
    {
      "<contract_title>": [
        {
          "category": "Governing Law",
          "question": "<full question text>",
          "spans": ["...annotated text...", ...],
          "span_starts": [482, ...]
        },
        ...41 entries per contract...
      ]
    }
    """
    annotations: dict[str, list] = {}

    for row in rows:
        title = row["title"]
        if title not in annotations:
            annotations[title] = []

        question = row["question"]
        match = re.search(r'related to "(.+?)" that should be', question)
        category_name = match.group(1) if match else "Unknown"

        annotations[title].append({
            "category": category_name,
            "question": question.strip(),
            "spans": row["answers"]["text"],
            "span_starts": row["answers"]["answer_start"],
        })

    out_path = DATA_DIR / "annotations.json"
    with open(out_path, "w") as f:
        json.dump(annotations, f, indent=2)

    total_spans = sum(
        len(entry["spans"])
        for entries in annotations.values()
        for entry in entries
    )
    print(f"  annotations.json: {len(annotations)} contracts, {total_spans} labeled spans")




def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_cuad()
    extract_taxonomy(rows)
    extract_contracts(rows)
    extract_annotations(rows)
    print("\nAll CUAD data saved to ../data/cuad/")

if __name__ == "__main__":
    main()
