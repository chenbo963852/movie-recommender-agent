from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)


class QdrantService:
    def __init__(self):
        self.client = QdrantClient(path="qdrant_data")
        self.collection_name = "documents"

    def recreate_collection(self, vector_size: int = 768):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            ),
        )

    def _to_vector_list(self, vector):
        """
        兼容 numpy array / list，确保写入 Qdrant 的 vector 是普通 list。
        """
        if hasattr(vector, "tolist"):
            return vector.tolist()
        return vector

    def _build_payload(self, doc: dict) -> dict:
        """
        保存完整 payload。
        新版 movie_documents.jsonl 里包含 title、overview、genres、keywords、
        cast、directors、year、runtime、vote_average 等字段，都要存进 Qdrant。
        """
        payload = dict(doc)

        # 兜底，防止旧数据或手动新增文档缺字段
        payload.setdefault("text", "")
        payload.setdefault("category", "Unknown")

        return payload

    def add_documents(self, docs: list[dict], vectors: list[list[float]]):
        points = []

        for doc, vector in zip(docs, vectors):
            point = PointStruct(
                id=int(doc["id"]),
                vector=self._to_vector_list(vector),
                payload=self._build_payload(doc)
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def add_document(self, doc: dict, vector: list[float]):
        point = PointStruct(
            id=int(doc["id"]),
            vector=self._to_vector_list(vector),
            payload=self._build_payload(doc)
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5
    ):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=self._to_vector_list(query_vector),
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        return response.points

    def get_documents(self, limit: int = 20):
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        return points

    def get_all_documents(self, batch_size: int = 500):
        all_points = []
        next_page_offset = None

        while True:
            points, next_page_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False
            )

            all_points.extend(points)

            if next_page_offset is None:
                break

        return all_points

    def get_document_by_id(self, doc_id: int):
        records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[int(doc_id)],
            with_payload=True,
            with_vectors=False
        )

        if not records:
            return None

        return records[0]
