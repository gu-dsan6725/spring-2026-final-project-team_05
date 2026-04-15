"""
Build ChromaDB vector store from CUAD contract files.

Run once before using the benchmark agent (python3 scripts/build_vector_store.py)

Vector store is saved to data/cuad_vector_store/ and loaded automatically by benchmark agent at runtime.
"""

import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

CONTRACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "contracts")
STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cuad_vector_store")
COLLECTION_NAME = "cuad_contracts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_CHUNK_CHARS = 150
MAX_CHUNK_CHARS = 2000
BATCH_SIZE = 500

def chunk_contract(text: str) -> list[str]:
    # Split contract text into paragraph level chunks of manageable size
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [p for p in paragraphs if MIN_CHUNK_CHARS <= len(p) <= MAX_CHUNK_CHARS]

def main():
    os.makedirs(STORE_DIR, exist_ok=True)

    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=STORE_DIR)

    # Delete & recreate collection for clean build
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, embedding_function=ef)

    contract_files = sorted(f for f in os.listdir(CONTRACTS_DIR) if f.endswith(".txt"))
    print(f"Found {len(contract_files)} CUAD contracts, building vector store...")

    batch_docs, batch_ids, batch_metas = [], [], []
    total_chunks = 0
    chunk_counter = 0

    for filename in tqdm(contract_files, desc="Indexing contracts"):
        filepath = os.path.join(CONTRACTS_DIR, filename)
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            tqdm.write(f"  Skipping {filename}: {e}")
            continue

        chunks = chunk_contract(text)
        for chunk in chunks:
            batch_docs.append(chunk)
            batch_ids.append(f"chunk_{chunk_counter}")
            batch_metas.append({"source": filename})
            chunk_counter += 1
            total_chunks += 1

            if len(batch_docs) >= BATCH_SIZE:
                collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
                batch_docs, batch_ids, batch_metas = [], [], []

    if batch_docs:
        collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)

    print("\nDone!")
    print(f"Indexed {total_chunks} chunks from {len(contract_files)} contracts")
    print(f"Vector store saved to: {os.path.abspath(STORE_DIR)}")

if __name__ == "__main__":
    main()
