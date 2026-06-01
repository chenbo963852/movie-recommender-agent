import json

from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.bm25_service import BM25Service


def main():
    print("Step 1: loading documents.json")
    with open("data/documents.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    print(f"Step 2: loaded {len(docs)} documents")

    embedding_service = EmbeddingService()
    qdrant_service = QdrantService()
    bm25_service = BM25Service()

    print("Step 3: recreate collection")
    qdrant_service.recreate_collection(vector_size=768)

    texts = [doc["text"] for doc in docs]

    print("Step 4: encoding documents")
    vectors = embedding_service.encode_batch(texts)

    print("Step 5: adding documents to Qdrant")
    qdrant_service.add_documents(docs, vectors)

    print("Step 6: building BM25 index")
    bm25_service.build_index(docs)

    print("Step 7: all movie data initialized successfully")


if __name__ == "__main__":
    main()
