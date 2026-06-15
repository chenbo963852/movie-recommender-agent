"""
Online evaluation helpers for the movie recommender.

Each helper holds out the user's most-recent highly-rated movies as
ground truth, runs the recommender on the remaining ratings, and
computes hit-rate / recall / NDCG at k.
"""

import math


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _split_user_ratings(
    ratings_service,
    user_id: int,
    holdout_count: int = 2,
    min_rating: float = 4.0,
):
    """
    Return (seed_ratings, held_out_movie_ids) for *user_id*.

    The held-out set is the *holdout_count* highest-rated movies.
    If ``timestamp`` is present in the raw CSV the ratings are sorted
    by recency first -- this keeps the online eval consistent with the
    offline time-aware split.
    """
    all_ratings = ratings_service.get_user_ratings(user_id)
    if not all_ratings:
        raise ValueError(f"User id {user_id} not found")

    liked = [r for r in all_ratings if r["rating"] >= min_rating]

    # Try time-aware sort; fall back to rating-then-movie_id.
    if liked and "timestamp" in liked[0]:
        liked.sort(key=lambda x: (x["timestamp"], x["rating"]), reverse=True)
    else:
        liked.sort(key=lambda x: (x["rating"], x["movie_id"]), reverse=True)

    held_out_ids = {r["movie_id"] for r in liked[:holdout_count]}
    seed_ratings = [r for r in all_ratings if r["movie_id"] not in held_out_ids]

    return seed_ratings, held_out_ids


def _compute_metrics(
    recommended_ids: list[int],
    ground_truth_ids: set[int],
    k: int,
) -> dict:
    top_k = recommended_ids[:k]

    hit_count = sum(1 for mid in top_k if mid in ground_truth_ids)
    hit_rate = 1.0 if hit_count > 0 else 0.0
    recall = safe_div(hit_count, len(ground_truth_ids))

    dcg = 0.0
    for rank, mid in enumerate(top_k, start=1):
        if mid in ground_truth_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hit_count = min(len(ground_truth_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hit_count + 1))
    ndcg = safe_div(dcg, idcg)

    return {
        "hit_rate": hit_rate,
        "recall": recall,
        "ndcg": ndcg,
        "hit_count": hit_count,
    }


def evaluate_recommendation(
    ratings_service,
    qdrant_service,
    embedding_service,
    recommend_for_user_core,
    user_id: int,
    top_k: int = 10,
    seed_limit: int = 5,
    min_rating: float = 4.0,
) -> dict:
    """
    Run a single-user online evaluation.

    1. Split the user's ratings into train / hold-out.
    2. Run the recommendation function.
    3. Report hit-rate, recall, NDCG @ k and a seen-violation check.
    """
    seed_ratings, ground_truth_ids = _split_user_ratings(
        ratings_service,
        user_id,
        holdout_count=2,
        min_rating=min_rating,
    )

    if len(ground_truth_ids) == 0:
        return {
            "user_id": user_id,
            "ground_truth_movie_count": 0,
            "metrics": {"hit_rate": 0.0, "recall": 0.0, "ndcg": 0.0, "hit_count": 0},
            "recommended_count": 0,
            "seed_count": 0,
            "note": "No hold-out movies available for this user.",
        }

    # Build a *temporary* ratings service that only sees the training portion
    # so the recommender doesn't cheat.
    from app.services.ratings_service import RatingsService
    from app.services.recommendation_service import UserRecommendationService

    holdout_ratings = RatingsService.__new__(RatingsService)
    holdout_ratings.movie_stats = ratings_service.movie_stats
    holdout_ratings.user_ratings = {user_id: seed_ratings}

    temp_recommender = UserRecommendationService(
        ratings_service=holdout_ratings,
        qdrant_service=qdrant_service,
        embedding_service=embedding_service,
        retrieval_service=None,
    )

    try:
        result = temp_recommender.recommend_for_user(
            user_id=user_id,
            top_k=top_k,
            seed_limit=seed_limit,
            min_rating=min_rating,
        )
    except Exception as exc:
        return {
            "user_id": user_id,
            "error": str(exc),
            "ground_truth_movie_count": len(ground_truth_ids),
            "metrics": None,
        }

    recommended_ids: list[int] = []
    for item in result.get("results", []):
        movie_id = item.get("id") or item.get("movie_id")
        if movie_id is not None:
            recommended_ids.append(int(movie_id))

    metrics = _compute_metrics(recommended_ids, ground_truth_ids, k=top_k)

    seed_ids = {r["movie_id"] for r in seed_ratings}
    recommended_set = set(recommended_ids[:top_k])
    seen_violation_count = len(recommended_set & seed_ids)

    return {
        "user_id": user_id,
        "ground_truth_movie_ids": sorted(ground_truth_ids),
        "ground_truth_movie_count": len(ground_truth_ids),
        "recommended_movie_ids": recommended_ids[:top_k],
        "recommended_count": len(recommended_ids[:top_k]),
        "seed_movie_count": len(seed_ids),
        "seen_violation_count": int(seen_violation_count),
        "seen_violation_rate": safe_div(seen_violation_count, max(top_k, 1)),
        "metrics": metrics,
    }


def evaluate_batch_users(
    ratings_service,
    qdrant_service,
    embedding_service,
    recommend_for_user_core,
    user_ids: list[int],
    top_k: int = 10,
    min_rating: float = 4.0,
    seed_limit: int = 5,
) -> dict:
    """
    Run ``evaluate_recommendation`` for multiple users and return
    aggregated metrics.
    """
    per_user = []
    total_hit = 0.0
    total_recall = 0.0
    total_ndcg = 0.0
    valid = 0

    for uid in user_ids:
        result = evaluate_recommendation(
            ratings_service=ratings_service,
            qdrant_service=qdrant_service,
            embedding_service=embedding_service,
            recommend_for_user_core=recommend_for_user_core,
            user_id=uid,
            top_k=top_k,
            seed_limit=seed_limit,
            min_rating=min_rating,
        )
        per_user.append(result)

        metrics = result.get("metrics")
        if metrics is not None:
            total_hit += metrics["hit_rate"]
            total_recall += metrics["recall"]
            total_ndcg += metrics["ndcg"]
            valid += 1

    return {
        "evaluated_users": valid,
        "total_users": len(user_ids),
        "hit_rate_at_k": safe_div(total_hit, valid),
        "recall_at_k": safe_div(total_recall, valid),
        "ndcg_at_k": safe_div(total_ndcg, valid),
        "top_k": top_k,
        "per_user": per_user,
    }
