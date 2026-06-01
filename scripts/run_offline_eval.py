import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService


EVAL_DIR = PROJECT_ROOT / "data/eval"
EVAL_PROCESSED_DIR = EVAL_DIR / "processed"
RESULTS_DIR = EVAL_DIR / "results"

TRAIN_RATINGS_PATH = EVAL_DIR / "ratings_train.csv"
GROUND_TRUTH_PATH = EVAL_DIR / "test_ground_truth.json"

MOVIE_RATING_STATS_PATH = EVAL_PROCESSED_DIR / "movie_rating_stats_train.parquet"
USER_SEEN_MOVIES_PATH = EVAL_PROCESSED_DIR / "user_seen_movies_train.parquet"

OUTPUT_JSON = RESULTS_DIR / "offline_eval_summary.json"
OUTPUT_CSV = RESULTS_DIR / "offline_eval_results.csv"


LIKED_RATING_THRESHOLD = 4.0
DEFAULT_TOP_K = 10
DEFAULT_SEED_LIMIT = 5


def safe_div(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def extract_year_from_text(text: str) -> int | None:
    if not text:
        return None

    match = re.search(r"\((\d{4})\)", text)
    if match:
        return int(match.group(1))

    return None


def get_payload_year(payload: dict) -> int | None:
    year = payload.get("year")

    if year is not None:
        try:
            return int(year)
        except Exception:
            pass

    text = payload.get("text") or payload.get("title") or ""
    return extract_year_from_text(text)


def load_ground_truth() -> dict[int, set[int]]:
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        int(user_id): set(map(int, movie_ids))
        for user_id, movie_ids in data.items()
    }


def load_user_seen_movies() -> dict[int, set[int]]:
    df = pd.read_parquet(USER_SEEN_MOVIES_PATH)

    result = {}

    for _, row in df.iterrows():
        user_id = int(row["userId"])
        seen_movie_ids = row["seen_movie_ids"]

        result[user_id] = set(map(int, seen_movie_ids))

    return result


def load_movie_rating_stats() -> dict[int, dict]:
    df = pd.read_parquet(MOVIE_RATING_STATS_PATH)

    result = {}

    for _, row in df.iterrows():
        movie_id = int(row["movieId"])
        rating_count = int(row["rating_count"])
        rating_mean = float(row["rating_mean"])

        result[movie_id] = {
            "rating_count": rating_count,
            "rating_mean": rating_mean,
            "popular_score": rating_mean * math.log1p(rating_count),
        }

    return result


def build_eval_user_train_ratings(
    train_ratings: pd.DataFrame,
    eval_user_ids: list[int],
) -> dict[int, list[dict]]:
    eval_user_set = set(eval_user_ids)

    df = train_ratings[train_ratings["userId"].isin(eval_user_set)].copy()

    has_timestamp = "timestamp" in df.columns

    result = {}

    for user_id, user_df in df.groupby("userId"):
        records = []

        for _, row in user_df.iterrows():
            item = {
                "movie_id": int(row["movieId"]),
                "rating": float(row["rating"]),
            }

            if has_timestamp:
                item["timestamp"] = int(row["timestamp"])

            records.append(item)

        result[int(user_id)] = records

    return result


def get_seed_movies(
    user_ratings: list[dict],
    seed_limit: int = DEFAULT_SEED_LIMIT,
    min_rating: float = LIKED_RATING_THRESHOLD,
) -> list[dict]:
    liked = [
        item
        for item in user_ratings
        if item["rating"] >= min_rating
    ]

    liked.sort(
        key=lambda x: (
            x["rating"],
            x.get("timestamp", 0),
        ),
        reverse=True,
    )

    return liked[:seed_limit]


def calculate_user_metrics(
    recommended_ids: list[int],
    ground_truth_ids: set[int],
    k: int,
) -> dict:
    top_k_ids = recommended_ids[:k]

    if not ground_truth_ids:
        return {
            "hit_rate": 0.0,
            "recall": 0.0,
            "ndcg": 0.0,
            "hit_count": 0,
        }

    hit_count = sum(1 for movie_id in top_k_ids if movie_id in ground_truth_ids)

    hit_rate = 1.0 if hit_count > 0 else 0.0
    recall = safe_div(hit_count, len(ground_truth_ids))

    dcg = 0.0
    for rank, movie_id in enumerate(top_k_ids, start=1):
        if movie_id in ground_truth_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hit_count = min(len(ground_truth_ids), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hit_count + 1)
    )

    ndcg = safe_div(dcg, idcg)

    return {
        "hit_rate": hit_rate,
        "recall": recall,
        "ndcg": ndcg,
        "hit_count": hit_count,
    }


class OfflineRecommender:
    def __init__(
        self,
        qdrant_service: QdrantService,
        embedding_service: EmbeddingService,
        movie_rating_stats: dict[int, dict],
        popular_movie_ids: list[int],
    ):
        self.qdrant_service = qdrant_service
        self.embedding_service = embedding_service
        self.movie_rating_stats = movie_rating_stats
        self.popular_movie_ids = popular_movie_ids

        self.document_cache = {}
        self.vector_cache = {}

    def get_document(self, movie_id: int):
        if movie_id in self.document_cache:
            return self.document_cache[movie_id]

        record = self.qdrant_service.get_document_by_id(movie_id)

        self.document_cache[movie_id] = record
        return record

    def get_embedding(self, movie_id: int, text: str):
        if movie_id in self.vector_cache:
            return self.vector_cache[movie_id]

        vector = self.embedding_service.encode(text)
        self.vector_cache[movie_id] = vector
        return vector

    def popular_baseline(
        self,
        seen_movie_ids: set[int],
        top_k: int,
    ) -> list[int]:
        results = []

        for movie_id in self.popular_movie_ids:
            if movie_id in seen_movie_ids:
                continue

            results.append(movie_id)

            if len(results) >= top_k:
                break

        return results

    def vector_seed(
        self,
        seed_movies: list[dict],
        seen_movie_ids: set[int],
        top_k: int,
    ) -> list[int]:
        candidate_map = {}

        for seed in seed_movies:
            seed_movie_id = int(seed["movie_id"])
            seed_rating = float(seed["rating"])

            target_movie = self.get_document(seed_movie_id)
            if target_movie is None:
                continue

            target_text = target_movie.payload.get("text") or target_movie.payload.get("title") or ""

            if not target_text:
                continue

            query_vector = self.get_embedding(seed_movie_id, target_text)

            candidates = self.qdrant_service.search(
                query_vector=query_vector,
                top_k=200,
            )

            for point in candidates:
                candidate_id = int(point.id)

                if candidate_id in seen_movie_ids:
                    continue

                if candidate_id == seed_movie_id:
                    continue

                vector_score = float(point.score)

                final_score = (
                    0.95 * vector_score
                    + 0.05 * min(seed_rating / 5.0, 1.0)
                )

                if (
                    candidate_id not in candidate_map
                    or final_score > candidate_map[candidate_id]
                ):
                    candidate_map[candidate_id] = final_score

        ranked = sorted(
            candidate_map.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [movie_id for movie_id, _ in ranked[:top_k]]

    def personalized_rerank(
        self,
        seed_movies: list[dict],
        seen_movie_ids: set[int],
        top_k: int,
    ) -> list[int]:
        candidate_map = {}

        for seed in seed_movies:
            seed_movie_id = int(seed["movie_id"])
            seed_rating = float(seed["rating"])

            target_movie = self.get_document(seed_movie_id)
            if target_movie is None:
                continue

            target_payload = target_movie.payload

            target_text = target_payload.get("text") or target_payload.get("title") or ""
            target_genres_text = target_payload.get("category") or ""
            target_genres = set(target_genres_text.lower().split())
            target_year = get_payload_year(target_payload)

            if not target_text:
                continue

            query_vector = self.get_embedding(seed_movie_id, target_text)

            candidates = self.qdrant_service.search(
                query_vector=query_vector,
                top_k=300,
            )

            for point in candidates:
                candidate_id = int(point.id)

                if candidate_id in seen_movie_ids:
                    continue

                if candidate_id == seed_movie_id:
                    continue

                candidate_payload = point.payload
                candidate_genres_text = candidate_payload.get("category") or ""
                candidate_genres = set(candidate_genres_text.lower().split())

                shared_genres = target_genres & candidate_genres
                genre_overlap_count = len(shared_genres)

                genre_penalty = 1.0 if genre_overlap_count > 0 else 0.2

                vector_score = float(point.score)

                candidate_year = get_payload_year(candidate_payload)

                year_bonus = 0.0
                if target_year is not None and candidate_year is not None:
                    year_gap = abs(target_year - candidate_year)

                    if year_gap <= 3:
                        year_bonus = 1.0
                    elif year_gap <= 10:
                        year_bonus = 0.5
                    elif year_gap <= 20:
                        year_bonus = 0.2

                stats = self.movie_rating_stats.get(candidate_id, {})
                rating_mean = stats.get("rating_mean") or 0.0
                rating_count = stats.get("rating_count") or 0

                ml_rating_score = min(rating_mean / 5.0, 1.0)
                ml_count_score = min(rating_count / 50000.0, 1.0)

                ml_quality_score = (
                    0.7 * ml_rating_score
                    + 0.3 * ml_count_score
                )

                final_score = (
                    0.45 * vector_score
                    + 0.25 * genre_overlap_count
                    + 0.10 * year_bonus
                    + 0.10 * min(seed_rating / 5.0, 1.0)
                    + 0.10 * ml_quality_score
                ) * genre_penalty

                if (
                    candidate_id not in candidate_map
                    or final_score > candidate_map[candidate_id]
                ):
                    candidate_map[candidate_id] = final_score

        ranked = sorted(
            candidate_map.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [movie_id for movie_id, _ in ranked[:top_k]]


def evaluate_method(
    method_name: str,
    eval_user_ids: list[int],
    ground_truth: dict[int, set[int]],
    user_seen_movies: dict[int, set[int]],
    user_train_ratings: dict[int, list[dict]],
    recommender: OfflineRecommender,
    top_k: int,
    seed_limit: int,
    total_movie_count: int,
) -> dict:
    total_hit_rate = 0.0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hit_count = 0
    total_seen_violation = 0

    valid_users = 0
    all_recommended_ids = []

    user_rows = []

    for idx, user_id in enumerate(eval_user_ids, start=1):
        gt_ids = ground_truth.get(user_id, set())
        seen_ids = user_seen_movies.get(user_id, set())
        ratings = user_train_ratings.get(user_id, [])

        if not gt_ids:
            continue

        seed_movies = get_seed_movies(
            ratings,
            seed_limit=seed_limit,
            min_rating=LIKED_RATING_THRESHOLD,
        )

        if method_name == "popular_baseline":
            recommended_ids = recommender.popular_baseline(
                seen_movie_ids=seen_ids,
                top_k=top_k,
            )
        elif method_name == "vector_seed":
            recommended_ids = recommender.vector_seed(
                seed_movies=seed_movies,
                seen_movie_ids=seen_ids,
                top_k=top_k,
            )
        elif method_name == "personalized_rerank":
            recommended_ids = recommender.personalized_rerank(
                seed_movies=seed_movies,
                seen_movie_ids=seen_ids,
                top_k=top_k,
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")

        metrics = calculate_user_metrics(
            recommended_ids=recommended_ids,
            ground_truth_ids=gt_ids,
            k=top_k,
        )

        seen_violation_count = len(set(recommended_ids[:top_k]) & seen_ids)

        total_hit_rate += metrics["hit_rate"]
        total_recall += metrics["recall"]
        total_ndcg += metrics["ndcg"]
        total_hit_count += metrics["hit_count"]
        total_seen_violation += seen_violation_count

        valid_users += 1
        all_recommended_ids.extend(recommended_ids[:top_k])

        user_rows.append(
            {
                "method": method_name,
                "user_id": user_id,
                "hit_rate_at_k": metrics["hit_rate"],
                "recall_at_k": metrics["recall"],
                "ndcg_at_k": metrics["ndcg"],
                "hit_count": metrics["hit_count"],
                "seen_violation_count": seen_violation_count,
                "recommended_count": len(recommended_ids[:top_k]),
                "ground_truth_count": len(gt_ids),
            }
        )

        if idx % 50 == 0:
            print(f"[{method_name}] evaluated {idx}/{len(eval_user_ids)} users")

    coverage = safe_div(len(set(all_recommended_ids)), total_movie_count)

    summary = {
        "method": method_name,
        "valid_users": valid_users,
        f"hit_rate_at_{top_k}": safe_div(total_hit_rate, valid_users),
        f"recall_at_{top_k}": safe_div(total_recall, valid_users),
        f"ndcg_at_{top_k}": safe_div(total_ndcg, valid_users),
        f"coverage_at_{top_k}": coverage,
        "total_hit_count": int(total_hit_count),
        "seen_violation_count": int(total_seen_violation),
        "seen_violation_rate": safe_div(
            total_seen_violation,
            valid_users * top_k,
        ),
    }

    return {
        "summary": summary,
        "user_rows": user_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed-limit", type=int, default=DEFAULT_SEED_LIMIT)
    parser.add_argument("--max-users", type=int, default=100)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth...")
    ground_truth = load_ground_truth()

    eval_user_ids = sorted(ground_truth.keys())

    if args.max_users is not None and args.max_users > 0:
        eval_user_ids = eval_user_ids[:args.max_users]

    print(f"Eval users: {len(eval_user_ids)}")

    print("Loading train ratings...")
    train_ratings = pd.read_csv(TRAIN_RATINGS_PATH)

    print("Loading user seen movies...")
    user_seen_movies = load_user_seen_movies()

    print("Loading movie rating stats...")
    movie_rating_stats = load_movie_rating_stats()

    print("Building user train ratings...")
    user_train_ratings = build_eval_user_train_ratings(
        train_ratings=train_ratings,
        eval_user_ids=eval_user_ids,
    )

    print("Building popular movie list...")
    popular_movie_ids = [
        movie_id
        for movie_id, _ in sorted(
            movie_rating_stats.items(),
            key=lambda x: x[1]["popular_score"],
            reverse=True,
        )
    ]

    total_movie_count = int(train_ratings["movieId"].nunique())

    print("Loading embedding model and Qdrant...")
    embedding_service = EmbeddingService()
    qdrant_service = QdrantService()

    recommender = OfflineRecommender(
        qdrant_service=qdrant_service,
        embedding_service=embedding_service,
        movie_rating_stats=movie_rating_stats,
        popular_movie_ids=popular_movie_ids,
    )

    methods = [
        "popular_baseline",
        "vector_seed",
        "personalized_rerank",
    ]

    all_summaries = []
    all_user_rows = []

    for method_name in methods:
        print(f"\nEvaluating method: {method_name}")

        result = evaluate_method(
            method_name=method_name,
            eval_user_ids=eval_user_ids,
            ground_truth=ground_truth,
            user_seen_movies=user_seen_movies,
            user_train_ratings=user_train_ratings,
            recommender=recommender,
            top_k=args.top_k,
            seed_limit=args.seed_limit,
            total_movie_count=total_movie_count,
        )

        all_summaries.append(result["summary"])
        all_user_rows.extend(result["user_rows"])

        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "top_k": args.top_k,
                "seed_limit": args.seed_limit,
                "max_users": args.max_users,
                "summaries": all_summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    pd.DataFrame(all_user_rows).to_csv(OUTPUT_CSV, index=False)

    print("\nDone.")
    print(f"Saved summary to: {OUTPUT_JSON}")
    print(f"Saved user-level results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
