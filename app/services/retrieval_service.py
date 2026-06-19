import time
import re
import logging


logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        embedding_service,
        qdrant_service,
        bm25_service,
        movie_rating_stats_service=None,
    ):
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.bm25_service = bm25_service
        self.movie_rating_stats_service = movie_rating_stats_service

    def normalize_score(self, score: float, max_score: float) -> float:
        if max_score <= 0:
            return 0.0
        return score / max_score

    def extract_display_title(self, text: str) -> str:
        if not text:
            return ""

        first_line = text.split("\n")[0]

        if first_line.startswith("Title: "):
            return first_line.replace("Title: ", "").strip()

        return first_line.strip()

    def parse_genre_param(self, value: str | None) -> list[str]:
        if not value:
            return []

        return [
            item.strip().lower()
            for item in value.split(",")
            if item.strip()
        ]

    def has_all_genres(self, movie_genres: str | None, required_genres: list[str]) -> bool:
        if not required_genres:
            return True

        if not movie_genres:
            return False

        movie_genres_lower = movie_genres.lower()

        return all(
            genre in movie_genres_lower
            for genre in required_genres
        )

    def has_excluded_genre(self, movie_genres: str | None, excluded_genres: list[str]) -> bool:
        if not excluded_genres:
            return False

        if not movie_genres:
            return False

        movie_genres_lower = movie_genres.lower()

        return any(
            genre in movie_genres_lower
            for genre in excluded_genres
        )

    def pass_movie_filters(
        self,
        item: dict,
        genre: str | None = None,
        exclude_genre: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        min_vote_average: float | None = None,
        min_vote_count: int | None = None,
    ) -> bool:
        genres_text = item.get("genres") or ""

        required_genres = self.parse_genre_param(genre)
        excluded_genres = self.parse_genre_param(exclude_genre)

        if not self.has_all_genres(genres_text, required_genres):
            return False

        if self.has_excluded_genre(genres_text, excluded_genres):
            return False

        year = item.get("year")
        if year is not None:
            try:
                year = int(year)
            except Exception:
                year = None

        if year_from is not None:
            if year is None or year < year_from:
                return False

        if year_to is not None:
            if year is None or year > year_to:
                return False

        vote_average = item.get("vote_average")
        if vote_average is not None:
            try:
                vote_average = float(vote_average)
            except Exception:
                vote_average = None

        if min_vote_average is not None:
            if vote_average is None or vote_average < min_vote_average:
                return False

        vote_count = item.get("vote_count")
        if vote_count is not None:
            try:
                vote_count = int(vote_count)
            except Exception:
                vote_count = None

        if min_vote_count is not None:
            if vote_count is None or vote_count < min_vote_count:
                return False

        return True

    def add_movie_lens_stats(self, item: dict) -> dict:
        if self.movie_rating_stats_service is not None:
            ml_stats = self.movie_rating_stats_service.get_stats(item.get("id"))

            item["ml_rating_count"] = ml_stats["ml_rating_count"]
            item["ml_rating_mean"] = ml_stats["ml_rating_mean"]
        else:
            item["ml_rating_count"] = 0
            item["ml_rating_mean"] = None

        return item

    def search_movies(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
        genre: str | None = None,
        exclude_genre: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        min_vote_average: float | None = None,
        min_vote_count: int | None = None,
    ):
        start_time = time.time()

        selected_genre = genre or category

        query_vector = self.embedding_service.encode(query)

        # 1. 向量召回，多拿一些，方便后面过滤
        vector_results = self.qdrant_service.search(
            query_vector,
            top_k=200
        )

        # 2. BM25 召回
        bm25_results = self.bm25_service.get_scores(query)
        bm25_results = bm25_results[:200]

        # 3. 分数归一化
        max_vector_score = max(
            [float(point.score) for point in vector_results],
            default=0.0
        )

        max_bm25_score = max(
            [float(item["bm25_score"]) for item in bm25_results],
            default=0.0
        )

        merged = {}

        # 4. 处理向量召回结果
        for point in vector_results:
            raw_vector_score = float(point.score)
            normalized_vector_score = self.normalize_score(
                raw_vector_score,
                max_vector_score
            )

            payload = point.payload
            text = payload.get("text", "")
            title = payload.get("title") or self.extract_display_title(text)

            merged[point.id] = {
                "id": point.id,
                "title": title,
                "genres": payload.get("category"),
                "overview": payload.get("overview"),
                "year": payload.get("year"),
                "vote_average": payload.get("vote_average"),
                "vote_count": payload.get("vote_count"),
                "popularity": payload.get("popularity"),
                "directors": payload.get("directors"),
                "cast": payload.get("cast"),
                "keywords": payload.get("keywords"),
                "vector_score": normalized_vector_score,
                "bm25_score": 0.0,
                "raw_vector_score": raw_vector_score,
                "raw_bm25_score": 0.0,
            }

        # 5. 处理 BM25 召回结果
        for item in bm25_results:
            doc_id = item["id"]

            raw_bm25_score = float(item["bm25_score"])
            normalized_bm25_score = self.normalize_score(
                raw_bm25_score,
                max_bm25_score
            )

            if doc_id in merged:
                merged[doc_id]["bm25_score"] = normalized_bm25_score
                merged[doc_id]["raw_bm25_score"] = raw_bm25_score
            else:
                # BM25 只返回 id/text/category，这里再从 Qdrant 取完整 payload
                record = self.qdrant_service.get_document_by_id(doc_id)

                if record is not None:
                    payload = record.payload
                    text = payload.get("text", "")
                    title = payload.get("title") or self.extract_display_title(text)

                    merged[doc_id] = {
                        "id": doc_id,
                        "title": title,
                        "genres": payload.get("category"),
                        "overview": payload.get("overview"),
                        "year": payload.get("year"),
                        "vote_average": payload.get("vote_average"),
                        "vote_count": payload.get("vote_count"),
                        "popularity": payload.get("popularity"),
                        "directors": payload.get("directors"),
                        "cast": payload.get("cast"),
                        "keywords": payload.get("keywords"),
                        "vector_score": 0.0,
                        "bm25_score": normalized_bm25_score,
                        "raw_vector_score": 0.0,
                        "raw_bm25_score": raw_bm25_score,
                    }
                else:
                    # BM25 结果不在 Qdrant 中（数据不一致的罕见情况）
                    # 用 text 提取尽可能多的元数据，避免后续过滤时因 None 被丢弃
                    text = item["text"]
                    title = self.extract_display_title(text)

                    # 尝试从标题提取年份
                    extracted_year = None
                    year_match = re.search(r"\((\d{4})\)", title)
                    if year_match:
                        extracted_year = int(year_match.group(1))

                    merged[doc_id] = {
                        "id": doc_id,
                        "title": title,
                        "genres": item.get("category"),
                        "overview": None,
                        "year": extracted_year,
                        "vote_average": 0,    # 用 0 而非 None，避免被质量过滤误杀
                        "vote_count": 0,
                        "popularity": 0,
                        "directors": None,
                        "cast": None,
                        "keywords": None,
                        "vector_score": 0.0,
                        "bm25_score": normalized_bm25_score,
                        "raw_vector_score": 0.0,
                        "raw_bm25_score": raw_bm25_score,
                    }

        # 6. 过滤 + 融合排序
        results = []

        for item in merged.values():
            if not self.pass_movie_filters(
                item,
                genre=selected_genre,
                exclude_genre=exclude_genre,
                year_from=year_from,
                year_to=year_to,
                min_vote_average=min_vote_average,
                min_vote_count=min_vote_count,
            ):
                continue

            # 加入 MovieLens 离线评分统计
            self.add_movie_lens_stats(item)

            vote_average = item.get("vote_average") or 0
            vote_count = item.get("vote_count") or 0
            popularity = item.get("popularity") or 0

            ml_rating_mean = item.get("ml_rating_mean") or 0
            ml_rating_count = item.get("ml_rating_count") or 0

            try:
                vote_average = float(vote_average)
            except Exception:
                vote_average = 0.0

            try:
                vote_count = int(vote_count)
            except Exception:
                vote_count = 0

            try:
                popularity = float(popularity)
            except Exception:
                popularity = 0.0

            try:
                ml_rating_mean = float(ml_rating_mean)
            except Exception:
                ml_rating_mean = 0.0

            try:
                ml_rating_count = int(ml_rating_count)
            except Exception:
                ml_rating_count = 0

            # TMDB quality, 0-1
            tmdb_rating_score = min(vote_average / 10, 1.0)
            tmdb_count_score = min(vote_count / 5000, 1.0)
            tmdb_popularity_score = min(popularity / 100, 1.0)

            tmdb_quality_score = (
                    0.5 * tmdb_rating_score
                    + 0.3 * tmdb_count_score
                    + 0.2 * tmdb_popularity_score
            )

            # MovieLens quality, 0-1
            # MovieLens rating is 0.5-5.0, so normalize by 5.
            ml_rating_score = min(ml_rating_mean / 5, 1.0)

            # 50000 is a soft cap for popularity/confidence in full MovieLens.
            ml_count_score = min(ml_rating_count / 50000, 1.0)

            ml_quality_score = (
                    0.7 * ml_rating_score
                    + 0.3 * ml_count_score
            )

            # Combined quality
            quality_score = (
                    0.6 * tmdb_quality_score
                    + 0.4 * ml_quality_score
            )

            final_score = (
                    0.55 * item["vector_score"]
                    + 0.25 * item["bm25_score"]
                    + 0.20 * quality_score
            )

            item["tmdb_quality_score"] = round(float(tmdb_quality_score), 4)
            item["ml_quality_score"] = round(float(ml_quality_score), 4)
            item["quality_score"] = round(float(quality_score), 4)
            item["final_score"] = round(float(final_score), 4)

            results.append(item)

        results.sort(key=lambda x: x["final_score"], reverse=True)
        final_results = results[:top_k]

        elapsed_time = time.time() - start_time

        logger.info(
            f"Search query='{query}', genre='{selected_genre}', "
            f"exclude_genre='{exclude_genre}', year_from={year_from}, year_to={year_to}, "
            f"min_vote_average={min_vote_average}, min_vote_count={min_vote_count}, top_k={top_k}, "
            f"result_count={len(final_results)}, elapsed_time={elapsed_time:.4f}s"
        )

        return {
            "query": query,
            "genre": selected_genre,
            "exclude_genre": exclude_genre,
            "year_from": year_from,
            "year_to": year_to,
            "min_vote_average": min_vote_average,
            "min_vote_count": min_vote_count,
            "results": final_results,
            "elapsed_time_seconds": round(elapsed_time, 4),
        }
