from app.services.qdrant_service import QdrantService
from app.services.embedding_service import EmbeddingService
import re


def extract_year(title: str) -> int | None:
    match = re.search(r"\((\d{4})\)", title or "")
    if match:
        return int(match.group(1))
    return None


def recommend_from_seed_movies(
    seed_movies: list[dict],
    watched_movie_ids: set[int],
    qdrant_service: QdrantService,
    embedding_service: EmbeddingService,
    top_k: int = 10,
    exclude_watched: bool = True,
) -> list[dict]:
    candidate_map = {}

    for seed in seed_movies:
        seed_movie_id = seed["movie_id"]
        seed_rating = seed["rating"]

        target_movie = qdrant_service.get_document_by_id(seed_movie_id)
        if target_movie is None:
            continue

        target_text = target_movie.payload.get("text", "")
        target_category = target_movie.payload.get("category", "")
        target_genres = set(target_category.lower().split())
        target_year = extract_year(target_text)

        query_vector = embedding_service.encode(target_text)

        candidates = qdrant_service.search(query_vector, top_k=300)

        for point in candidates:
            candidate_id = int(point.id)

            if candidate_id == seed_movie_id:
                continue

            if exclude_watched and candidate_id in watched_movie_ids:
                continue

            candidate_title = point.payload.get("text", "")
            candidate_genres_text = point.payload.get("category", "")
            candidate_genres = set(candidate_genres_text.lower().split())

            shared_genres = target_genres & candidate_genres
            genre_overlap_count = len(shared_genres)
            genre_penalty = 1.0 if genre_overlap_count > 0 else 0.2

            vector_score = float(point.score)

            candidate_year = extract_year(candidate_title)
            year_bonus = 0.0
            year_gap = None

            if target_year is not None and candidate_year is not None:
                year_gap = abs(target_year - candidate_year)
                if year_gap <= 3:
                    year_bonus = 1.0
                elif year_gap <= 10:
                    year_bonus = 0.5
                elif year_gap <= 20:
                    year_bonus = 0.2

            final_score = (
                0.45 * vector_score
                + 0.35 * genre_overlap_count
                + 0.15 * year_bonus
                + 0.05 * seed_rating
                + 0.05 * (1.0 if seed_rating >= 4.5 else 0.0)
            ) * genre_penalty

            if (
                candidate_id not in candidate_map
                or final_score > candidate_map[candidate_id]["final_score"]
            ):
                candidate_map[candidate_id] = {
                    "id": candidate_id,
                    "title": candidate_title,
                    "genres": candidate_genres_text,
                    "vector_score": vector_score,
                    "genre_overlap_count": genre_overlap_count,
                    "shared_genres": sorted(shared_genres),
                    "candidate_year": candidate_year,
                    "year_gap": year_gap,
                    "year_bonus": year_bonus,
                    "source_seed_movie_id": seed_movie_id,
                    "source_seed_title": target_text,
                    "source_seed_genres": target_category,
                    "source_seed_rating": seed_rating,
                    "final_score": float(final_score),
                }

    results = list(candidate_map.values())
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_k]


class UserRecommendationService:
    def __init__(
            self,
            ratings_service,
            qdrant_service: QdrantService,
            embedding_service: EmbeddingService,
            retrieval_service=None,
    ):
        self.ratings_service = ratings_service
        self.qdrant_service = qdrant_service
        self.embedding_service = embedding_service
        self.retrieval_service = retrieval_service

    def get_user_seed_movies(
        self,
        user_id: int,
        min_rating: float = 4.0,
        max_seeds: int = 5,
        excluded_movie_ids: set[int] | None = None,
    ) -> list[dict]:
        ratings = self.ratings_service.get_user_ratings(user_id)

        high_rated = [
            item
            for item in ratings
            if item["rating"] >= min_rating
            and (
                excluded_movie_ids is None
                or item["movie_id"] not in excluded_movie_ids
            )
        ]

        high_rated.sort(key=lambda x: x["rating"], reverse=True)

        return high_rated[:max_seeds]

    def get_user_watched_movie_ids(
        self,
        user_id: int,
        excluded_movie_ids: set[int] | None = None,
    ) -> set[int]:
        ratings = self.ratings_service.get_user_ratings(user_id)

        watched = {int(item["movie_id"]) for item in ratings}

        if excluded_movie_ids:
            watched -= set(map(int, excluded_movie_ids))

        return watched

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 20,
        min_rating: float = 4.0,
        seed_limit: int = 10,
        excluded_movie_ids: set[int] | None = None,
        exclude_watched: bool = True,
    ) -> dict:
        if not self.ratings_service.is_loaded():
            raise ValueError("Ratings data not loaded")

        user_ratings = self.ratings_service.get_user_ratings(user_id)
        if not user_ratings:
            raise ValueError(f"User id {user_id} not found")

        seed_movies = self.get_user_seed_movies(
            user_id=user_id,
            min_rating=min_rating,
            max_seeds=seed_limit,
            excluded_movie_ids=excluded_movie_ids,
        )

        if not seed_movies:
            return {
                "user_id": user_id,
                "seed_movies": [],
                "results": [],
            }

        seed_movie_details = []

        for seed in seed_movies:
            seed_movie = self.qdrant_service.get_document_by_id(seed["movie_id"])
            if seed_movie is not None:
                seed_movie_details.append(
                    {
                        "movie_id": seed["movie_id"],
                        "title": seed_movie.payload.get("text"),
                        "genres": seed_movie.payload.get("category"),
                        "rating": seed["rating"],
                    }
                )

        watched_movie_ids = self.get_user_watched_movie_ids(
            user_id=user_id,
            excluded_movie_ids=excluded_movie_ids,
        )

        results = recommend_from_seed_movies(
            seed_movies=seed_movies,
            watched_movie_ids=watched_movie_ids,
            qdrant_service=self.qdrant_service,
            embedding_service=self.embedding_service,
            top_k=top_k,
            exclude_watched=exclude_watched,
        )

        return {
            "user_id": user_id,
            "seed_movies": seed_movie_details,
            "results": results,
        }

    def recommend_for_user_hybrid(
            self,
            user_id: int,
            top_k: int = 20,
            min_rating: float = 4.0,
            seed_limit: int = 5,
            excluded_movie_ids: set[int] | None = None,
            exclude_watched: bool = True,
    ) -> dict:
        if self.retrieval_service is None:
            raise ValueError("retrieval_service is not available")

        if not self.ratings_service.is_loaded():
            raise ValueError("Ratings data not loaded")

        user_ratings = self.ratings_service.get_user_ratings(user_id)
        if not user_ratings:
            raise ValueError(f"User id {user_id} not found")

        seed_movies = self.get_user_seed_movies(
            user_id=user_id,
            min_rating=min_rating,
            max_seeds=seed_limit,
            excluded_movie_ids=excluded_movie_ids,
        )

        if not seed_movies:
            return {
                "user_id": user_id,
                "seed_movies": [],
                "results": [],
            }

        watched_movie_ids = self.get_user_watched_movie_ids(
            user_id=user_id,
            excluded_movie_ids=excluded_movie_ids,
        )

        seed_movie_details = []
        candidate_map = {}

        for seed in seed_movies:
            seed_movie_id = int(seed["movie_id"])
            seed_rating = float(seed["rating"])

            seed_movie = self.qdrant_service.get_document_by_id(seed_movie_id)
            if seed_movie is None:
                continue

            seed_payload = seed_movie.payload
            seed_title = seed_payload.get("title") or seed_payload.get("text", "")
            seed_text = seed_payload.get("text", "") or seed_title
            seed_genres_text = seed_payload.get("category") or ""

            seed_movie_details.append(
                {
                    "movie_id": seed_movie_id,
                    "title": seed_title,
                    "genres": seed_genres_text,
                    "rating": seed_rating,
                }
            )

            search_result = self.retrieval_service.search_movies(
                query=seed_text,
                top_k=200,
                genre=None,
                exclude_genre=None,
                year_from=None,
                year_to=None,
                min_vote_average=None,
                min_vote_count=None,
            )

            seed_genres = set(seed_genres_text.lower().split())

            for movie in search_result["results"]:
                candidate_id = int(movie["id"])

                if candidate_id == seed_movie_id:
                    continue

                if exclude_watched and candidate_id in watched_movie_ids:
                    continue

                movie_genres_text = movie.get("genres") or ""
                movie_genres = set(movie_genres_text.lower().split())

                shared_genres = seed_genres & movie_genres
                genre_overlap_count = len(shared_genres)

                genre_penalty = 1.0 if genre_overlap_count > 0 else 0.2

                retrieval_score = float(movie.get("final_score", 0.0) or 0.0)
                quality_score = float(movie.get("quality_score", 0.0) or 0.0)
                ml_rating_count = int(movie.get("ml_rating_count", 0) or 0)

                user_score = min(seed_rating / 5.0, 1.0)

                final_score = (
                                      0.55 * retrieval_score
                                      + 0.20 * genre_overlap_count
                                      + 0.15 * quality_score
                                      + 0.10 * user_score
                              ) * genre_penalty

                movie["source_seed_movie_id"] = seed_movie_id
                movie["source_seed_title"] = seed_title
                movie["source_seed_rating"] = seed_rating
                movie["genre_overlap_count"] = genre_overlap_count
                movie["shared_genres"] = sorted(shared_genres)
                movie["hybrid_user_score"] = round(float(final_score), 4)
                movie["ml_rating_count"] = ml_rating_count

                if (
                        candidate_id not in candidate_map
                        or final_score > candidate_map[candidate_id]["hybrid_user_score"]
                ):
                    candidate_map[candidate_id] = movie

        results = list(candidate_map.values())

        results.sort(
            key=lambda x: (
                x.get("genre_overlap_count", 0),
                x.get("hybrid_user_score", 0),
                x.get("ml_rating_count", 0),
            ),
            reverse=True,
        )

        return {
            "user_id": user_id,
            "seed_movies": seed_movie_details,
            "results": results[:top_k],
        }
