import json
import math
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.ratings_service import RatingsService
from app.services.recommendation_service import UserRecommendationService
from app.services.bm25_service import BM25Service
from app.services.retrieval_service import RetrievalService
from app.services.movie_rating_stats_service import MovieRatingStatsService



EVAL_DIR = PROJECT_ROOT / "data/eval"
RESULTS_DIR = EVAL_DIR / "results"

TRAIN_RATINGS_PATH = EVAL_DIR / "ratings_train.csv"
GROUND_TRUTH_PATH = EVAL_DIR / "test_ground_truth.json"

OUTPUT_SUMMARY = RESULTS_DIR / "main_hybrid_recommender_pipeline.json"
OUTPUT_RESULTS = RESULTS_DIR / "main_hybrid_recommender_pipeline.csv"


TOP_K = 10
MIN_RATING = 4.0
SEED_LIMIT = 5
MAX_USERS = 1000


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return round(numerator / denominator, 6)


def calculate_metrics(
    recommended_ids: list[int],
    ground_truth_ids: set[int],
    k: int,
):
    top_k_ids = recommended_ids[:k]

    hit_count = sum(
        1 for movie_id in top_k_ids
        if movie_id in ground_truth_ids
    )

    hit_rate = 1.0 if hit_count > 0 else 0.0

    recall = safe_div(
        hit_count,
        len(ground_truth_ids),
    )

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


def load_ground_truth():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        int(user_id): set(map(int, movie_ids))
        for user_id, movie_ids in data.items()
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth...")

    ground_truth = load_ground_truth()

    eval_user_ids = sorted(ground_truth.keys())[:MAX_USERS]

    print(f"Eval users: {len(eval_user_ids)}")

    print("Loading train-only ratings service...")

    train_ratings_service = RatingsService(
        ratings_file=str(TRAIN_RATINGS_PATH)
    )

    train_ratings_service.load_ratings()

    print("Loading embedding service...")
    embedding_service = EmbeddingService()

    print("Loading Qdrant service...")
    qdrant_service = QdrantService()

    print("Building UserRecommendationService...")

    print("Loading BM25 service...")
    bm25_service = BM25Service()
    if not bm25_service.load_cache_if_exists():
        raise RuntimeError("BM25 cache not found. Please start app once or rebuild BM25 index.")

    print("Loading train-only movie rating stats...")
    movie_rating_stats_service = MovieRatingStatsService()
    movie_rating_stats_service.load_stats(
        path="data/eval/processed/movie_rating_stats_train.parquet"
    )

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        bm25_service=bm25_service,
        movie_rating_stats_service=movie_rating_stats_service,
    )

    user_recommendation_service = UserRecommendationService(
        ratings_service=train_ratings_service,
        qdrant_service=qdrant_service,
        embedding_service=embedding_service,
        retrieval_service=retrieval_service,
    )

    total_hit_rate = 0.0
    total_recall = 0.0
    total_ndcg = 0.0

    total_hit_count = 0
    total_seen_violation_count = 0

    all_recommended_movie_ids = []

    user_rows = []

    for idx, user_id in enumerate(eval_user_ids, start=1):
        ground_truth_ids = ground_truth[user_id]

        try:
            result = user_recommendation_service.recommend_for_user_hybrid(
                user_id=user_id,
                top_k=TOP_K,
                min_rating=MIN_RATING,
                seed_limit=SEED_LIMIT,
                excluded_movie_ids=ground_truth_ids,
                exclude_watched=True,
            )

            recommended_ids = [
                int(item["id"])
                for item in result["results"]
            ]

            seed_movie_ids = {
                int(item["movie_id"])
                for item in result["seed_movies"]
            }

            seen_violation_count = len(
                set(recommended_ids[:TOP_K]) & seed_movie_ids
            )

            metrics = calculate_metrics(
                recommended_ids=recommended_ids,
                ground_truth_ids=ground_truth_ids,
                k=TOP_K,
            )

            total_hit_rate += metrics["hit_rate"]
            total_recall += metrics["recall"]
            total_ndcg += metrics["ndcg"]

            total_hit_count += metrics["hit_count"]
            total_seen_violation_count += seen_violation_count

            all_recommended_movie_ids.extend(
                recommended_ids[:TOP_K]
            )

            user_rows.append(
                {
                    "user_id": user_id,
                    "hit_rate_at_10": metrics["hit_rate"],
                    "recall_at_10": metrics["recall"],
                    "ndcg_at_10": metrics["ndcg"],
                    "hit_count": metrics["hit_count"],
                    "recommended_count": len(recommended_ids[:TOP_K]),
                    "ground_truth_count": len(ground_truth_ids),
                    "seen_violation_count": seen_violation_count,
                }
            )

        except Exception as e:
            print(f"User {user_id} failed: {e}")

        if idx % 50 == 0:
            print(f"Evaluated {idx}/{len(eval_user_ids)} users")

    coverage = safe_div(
        len(set(all_recommended_movie_ids)),
        train_ratings_service.movie_stats.__len__(),
    )

    summary = {
        "model": "main_recommender_pipeline",
        "top_k": TOP_K,
        "seed_limit": SEED_LIMIT,
        "min_rating": MIN_RATING,
        "eval_users": len(eval_user_ids),

        "hit_rate_at_10": safe_div(
            total_hit_rate,
            len(eval_user_ids),
        ),

        "recall_at_10": safe_div(
            total_recall,
            len(eval_user_ids),
        ),

        "ndcg_at_10": safe_div(
            total_ndcg,
            len(eval_user_ids),
        ),

        "coverage_at_10": coverage,

        "total_hit_count": int(total_hit_count),

        "seen_violation_count": int(
            total_seen_violation_count
        ),

        "seen_violation_rate": safe_div(
            total_seen_violation_count,
            len(eval_user_ids) * TOP_K,
        ),
    }

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    pd.DataFrame(user_rows).to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    print("\nDone.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\nSaved summary to: {OUTPUT_SUMMARY}")
    print(f"Saved results to: {OUTPUT_RESULTS}")


if __name__ == "__main__":
    main()
