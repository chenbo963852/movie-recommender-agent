import json
import random
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_RATINGS_PATH = PROJECT_ROOT / "data/raw/ratings.csv"
EVAL_DIR = PROJECT_ROOT / "data/eval"

TRAIN_OUTPUT = EVAL_DIR / "ratings_train.csv"
TEST_OUTPUT = EVAL_DIR / "ratings_test.csv"
GROUND_TRUTH_OUTPUT = EVAL_DIR / "test_ground_truth.json"
SUMMARY_OUTPUT = EVAL_DIR / "eval_summary.json"


RANDOM_SEED = 42

MIN_USER_RATINGS = 20
MIN_USER_LIKED = 5
LIKED_RATING_THRESHOLD = 4.0

TEST_LIKED_PER_USER = 2
MAX_EVAL_USERS = 1000


def main():
    random.seed(RANDOM_SEED)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading ratings from: {RAW_RATINGS_PATH}")

    ratings = pd.read_csv(RAW_RATINGS_PATH)

    required_cols = {"userId", "movieId", "rating"}
    missing_cols = required_cols - set(ratings.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in ratings.csv: {missing_cols}")

    has_timestamp = "timestamp" in ratings.columns

    print(f"Total ratings: {len(ratings)}")
    print(f"Total users: {ratings['userId'].nunique()}")
    print(f"Total movies: {ratings['movieId'].nunique()}")

    user_stats = (
        ratings.groupby("userId")
        .agg(
            rating_count=("rating", "count"),
            liked_count=("rating", lambda x: int((x >= LIKED_RATING_THRESHOLD).sum())),
        )
        .reset_index()
    )

    eligible_users = user_stats[
        (user_stats["rating_count"] >= MIN_USER_RATINGS)
        & (user_stats["liked_count"] >= MIN_USER_LIKED)
    ]["userId"].tolist()

    random.shuffle(eligible_users)
    eval_users = eligible_users[:MAX_EVAL_USERS]

    print(f"Eligible users: {len(eligible_users)}")
    print(f"Selected eval users: {len(eval_users)}")

    train_parts = []
    test_parts = []
    ground_truth = {}

    eval_user_set = set(eval_users)

    # 非评估用户的评分全部放入 train，保持整体电影统计更稳定
    non_eval_ratings = ratings[~ratings["userId"].isin(eval_user_set)].copy()
    train_parts.append(non_eval_ratings)

    eval_ratings = ratings[ratings["userId"].isin(eval_user_set)].copy()

    for user_id, user_df in eval_ratings.groupby("userId"):
        user_df = user_df.copy()

        liked_df = user_df[user_df["rating"] >= LIKED_RATING_THRESHOLD].copy()

        if len(liked_df) < TEST_LIKED_PER_USER:
            train_parts.append(user_df)
            continue

        if has_timestamp:
            liked_df = liked_df.sort_values("timestamp", ascending=False)
            test_df = liked_df.head(TEST_LIKED_PER_USER)
        else:
            test_df = liked_df.sample(
                n=TEST_LIKED_PER_USER,
                random_state=RANDOM_SEED,
            )

        test_indices = set(test_df.index)

        train_df = user_df[~user_df.index.isin(test_indices)].copy()

        # 防止极端情况：训练历史太少
        if len(train_df) < MIN_USER_RATINGS - TEST_LIKED_PER_USER:
            train_parts.append(user_df)
            continue

        train_parts.append(train_df)
        test_parts.append(test_df)

        ground_truth[str(int(user_id))] = [
            int(movie_id) for movie_id in test_df["movieId"].tolist()
        ]

    train_ratings = pd.concat(train_parts, ignore_index=True)
    test_ratings = pd.concat(test_parts, ignore_index=True)

    train_ratings.to_csv(TRAIN_OUTPUT, index=False)
    test_ratings.to_csv(TEST_OUTPUT, index=False)

    with open(GROUND_TRUTH_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    summary = {
        "random_seed": RANDOM_SEED,
        "liked_rating_threshold": LIKED_RATING_THRESHOLD,
        "min_user_ratings": MIN_USER_RATINGS,
        "min_user_liked": MIN_USER_LIKED,
        "test_liked_per_user": TEST_LIKED_PER_USER,
        "max_eval_users": MAX_EVAL_USERS,
        "selected_eval_users": len(eval_users),
        "valid_eval_users_with_test": len(ground_truth),
        "train_rating_count": int(len(train_ratings)),
        "test_rating_count": int(len(test_ratings)),
        "train_user_count": int(train_ratings["userId"].nunique()),
        "test_user_count": int(test_ratings["userId"].nunique()),
        "train_movie_count": int(train_ratings["movieId"].nunique()),
        "test_movie_count": int(test_ratings["movieId"].nunique()),
        "has_timestamp": has_timestamp,
        "outputs": {
            "train": str(TRAIN_OUTPUT),
            "test": str(TEST_OUTPUT),
            "ground_truth": str(GROUND_TRUTH_OUTPUT),
        },
    }

    with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
