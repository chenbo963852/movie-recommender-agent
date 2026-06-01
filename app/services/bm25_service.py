import json
import pickle
import hashlib
import re
from pathlib import Path
from rank_bm25 import BM25Okapi


class BM25Service:
    def __init__(self, cache_dir: str = "cache"):
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.index_cache_path = self.cache_dir / "bm25_index.pkl"
        self.meta_cache_path = self.cache_dir / "bm25_meta.json"

    def _get_documents_hash(self, documents: list[dict]) -> str:
        """
        用文档内容生成 hash。
        如果文档没变，下次就直接加载缓存。
        """
        content = json.dumps(
            [
                {
                    "id": doc.get("id"),
                    "text": doc.get("text", ""),
                    "category": doc.get("category", "")
                }
                for doc in documents
            ],
            ensure_ascii=False,
            sort_keys=True
        )
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def build_index(self, documents: list[dict], force_rebuild: bool = False):
        documents_hash = self._get_documents_hash(documents)

        if not force_rebuild and self._load_cache(documents_hash):
            print(f"BM25 index loaded from cache with {len(self.documents)} documents.")
            return

        print("BM25 cache not found or documents changed, rebuilding index...")

        self.documents = documents

        self.tokenized_corpus = []
        for doc in self.documents:
            searchable_text = " ".join([
                self._field_to_text(doc.get("title", "")),
                self._field_to_text(doc.get("text", "")),
                self._field_to_text(doc.get("overview", "")),
                self._field_to_text(doc.get("category", "")),
                self._field_to_text(doc.get("genres", "")),
                self._field_to_text(doc.get("keywords", "")),
                self._field_to_text(doc.get("directors", "")),
                self._field_to_text(doc.get("cast", "")),
            ])

            self.tokenized_corpus.append(self._tokenize(searchable_text))

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self._save_cache(documents_hash)

        print(f"BM25 index rebuilt with {len(self.documents)} documents.")

    def _save_cache(self, documents_hash: str):
        with open(self.index_cache_path, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "tokenized_corpus": self.tokenized_corpus,
                "bm25": self.bm25
            }, f)

        with open(self.meta_cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "documents_hash": documents_hash,
                "documents_count": len(self.documents)
            }, f, ensure_ascii=False, indent=2)

    def _load_cache(self, documents_hash: str) -> bool:
        if not self.index_cache_path.exists():
            return False

        if not self.meta_cache_path.exists():
            return False

        try:
            with open(self.meta_cache_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if meta.get("documents_hash") != documents_hash:
                return False

            with open(self.index_cache_path, "rb") as f:
                data = pickle.load(f)

            self.documents = data["documents"]
            self.tokenized_corpus = data["tokenized_corpus"]
            self.bm25 = data["bm25"]

            return True

        except Exception as e:
            print(f"Failed to load BM25 cache: {e}")
            return False

    def load_cache_if_exists(self) -> bool:
        """
        启动时直接加载 BM25 缓存。
        不检查 documents_hash，避免启动时还要读取全部文档。
        """
        if not self.index_cache_path.exists():
            return False

        try:
            with open(self.index_cache_path, "rb") as f:
                data = pickle.load(f)

            self.documents = data["documents"]
            self.tokenized_corpus = data["tokenized_corpus"]
            self.bm25 = data["bm25"]

            print(f"BM25 index loaded from cache with {len(self.documents)} documents.")
            return True

        except Exception as e:
            print(f"Failed to load BM25 cache: {e}")
            return False

    def get_scores(self, query: str, top_k: int = 1000):
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        results = []
        for doc, score in zip(self.documents, scores):
            if score <= 0:
                continue

            results.append({
                "id": doc["id"],
                "text": doc.get("text", ""),
                "category": doc.get("category", doc.get("genres", "")),
                "bm25_score": float(score)
            })

        results.sort(key=lambda x: x["bm25_score"], reverse=True)

        return results[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        return re.findall(r"[a-zA-Z0-9]+", text.lower()) # 连续的英文字母或数字进行分词

    def _field_to_text(self, value) -> str:
        if value is None:
            return ""

        if isinstance(value, list):
            return " ".join(self._field_to_text(item) for item in value)

        if isinstance(value, dict):
            return " ".join(self._field_to_text(v) for v in value.values())

        return str(value)

