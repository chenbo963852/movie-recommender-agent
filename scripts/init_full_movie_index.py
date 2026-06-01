import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService


INPUT_FILE = Path("data/processed/movie_documents.jsonl")
BATCH_SIZE = 512
VECTOR_SIZE = 768


def load_documents():
    docs = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    return docs


def main():
    print("Step 1: loading movie documents")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    embedding_service = EmbeddingService()
    qdrant_service = QdrantService()

    print("Step 2: recreating Qdrant collection")
    qdrant_service.recreate_collection(vector_size=VECTOR_SIZE)

    print("Step 3: encoding and inserting documents in batches")

    total = len(docs)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_docs = docs[start:end]

        texts = [doc["text"] for doc in batch_docs]
        vectors = embedding_service.encode_batch(texts)

        qdrant_service.add_documents(batch_docs, vectors)

        print(f"Inserted {end}/{total}")

    print("Done. Full movie index initialized successfully.")


if __name__ == "__main__":
    main()
