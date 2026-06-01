import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_RATINGS_PATH = PROJECT_ROOT / "data/eval/ratings_train.csv"
MOVIE_DOCS_PATH = PROJECT_ROOT / "data/processed/movie_documents.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "data/eval/processed"

MOVIE_RATING_STATS_OUTPUT = OUTPUT_DIR / "movie_rating_stats_train.parquet"
USER_PROFILES_OUTPUT = OUTPUT_DIR / "user_profiles_train.parquet"
USER_SEEN_MOVIES_OUTPUT = OUTPUT_DIR / "user_seen_movies_train.parquet"

MIN_LIKED_RATING = 4.0


def load_movie_genres() -> pd.DataFrame:
    rows = []

    with open(MOVIE_DOCS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            doc = json.loads(line)

            movie_id = doc.get("movie_id") or doc.get("id")
            genres = doc.get("genres") or []

            if movie_id is None:
                continue

            if isinstance(genres, str):
                genres = genres.split()

            for genre in genres:
                if genre:
                    rows.append(
                        {
                            "movieId": int(movie_id),
                            "genre": str(genre),
                        }
                    )

    df = pd.DataFrame(rows)

    print(f"Loaded genres for {df['movieId'].nunique()} movies.")
    return df


def build_movie_rating_stats(train_ratings: pd.DataFrame):
    print("Building train-only movie rating stats...")

    stats = (
        train_ratings.groupby("movieId")
        .agg(
            rating_count=("rating", "count"),
            rating_mean=("rating", "mean"),
        )
        .reset_index()
        .sort_values("rating_count", ascending=False)
    )

    stats.to_parquet(MOVIE_RATING_STATS_OUTPUT, index=False)

    print(f"Saved: {MOVIE_RATING_STATS_OUTPUT}")
    print(f"Movie stats count: {len(stats)}")
    print(stats.head(10))


def build_user_seen_movies(train_ratings: pd.DataFrame):
    print("Building train-only user seen movies...")

    df = train_ratings[["userId", "movieId"]].drop_duplicates()

    user_seen = (
        df.groupby("userId")["movieId"]
        .apply(lambda x: list(map(int, x.tolist())))
        .reset_index(name="seen_movie_ids")
    )

    user_seen.to_parquet(USER_SEEN_MOVIES_OUTPUT, index=False)

    print(f"Saved: {USER_SEEN_MOVIES_OUTPUT}")
    print(f"User seen count: {len(user_seen)}")
    print(user_seen.head(10))


def build_user_profiles(train_ratings: pd.DataFrame):
    print("Building train-only user profiles...")

    movie_genres_df = load_movie_genres()

    user_stats = (
        train_ratings.groupby("userId")
        .agg(
            rating_count=("rating", "count"),
            rating_sum=("rating", "sum"),
        )
        .reset_index()
    )

    user_stats["avg_rating"] = (
        user_stats["rating_sum"] / user_stats["rating_count"]
    )

    liked = train_ratings[train_ratings["rating"] >= MIN_LIKED_RATING].copy()

    liked_stats = (
        liked.groupby("userId")
        .agg(
            liked_count=("rating", "count"),
        )
        .reset_index()
    )

    liked_with_genres = liked.merge(
        movie_genres_df,
        on="movieId",
        how="inner",
    )

    genre_prefs = (
        liked_with_genres.groupby(["userId", "genre"])
        .size()
        .reset_index(name="genre_count")
    )

    genre_prefs = genre_prefs.sort_values(
        ["userId", "genre_count"],
        ascending=[True, False],
    )

    top_genres = genre_prefs.groupby("userId").head(5)

    preferred_genres = (
        top_genres.groupby("userId")["genre"]
        .apply(lambda x: ",".join(x.tolist()))
        .reset_index(name="preferred_genres")
    )

    profiles = user_stats.merge(
        liked_stats,
        on="userId",
        how="left",
    )

    profiles = profiles.merge(
        preferred_genres,
        on="userId",
        how="left",
    )

    profiles["liked_count"] = profiles["liked_count"].fillna(0).astype(int)
    profiles["preferred_genres"] = profiles["preferred_genres"].fillna("")

    profiles = profiles[
        [
            "userId",
            "rating_count",
            "liked_count",
            "avg_rating",
            "preferred_genres",
        ]
    ]

    profiles = profiles.sort_values("rating_count", ascending=False)

    profiles.to_parquet(USER_PROFILES_OUTPUT, index=False)

    print(f"Saved: {USER_PROFILES_OUTPUT}")
    print(f"User profile count: {len(profiles)}")
    print(profiles.head(10))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading train ratings from: {TRAIN_RATINGS_PATH}")

    if not TRAIN_RATINGS_PATH.exists():
        raise FileNotFoundError(f"Train ratings not found: {TRAIN_RATINGS_PATH}")

    if not MOVIE_DOCS_PATH.exists():
        raise FileNotFoundError(f"Movie documents not found: {MOVIE_DOCS_PATH}")

    train_ratings = pd.read_csv(TRAIN_RATINGS_PATH)

    required_cols = {"userId", "movieId", "rating"}
    missing_cols = required_cols - set(train_ratings.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"Train ratings count: {len(train_ratings)}")
    print(f"Train users count: {train_ratings['userId'].nunique()}")
    print(f"Train movies count: {train_ratings['movieId'].nunique()}")

    build_movie_rating_stats(train_ratings)
    build_user_seen_movies(train_ratings)
    build_user_profiles(train_ratings)

    summary = {
        "train_ratings_count": int(len(train_ratings)),
        "train_users_count": int(train_ratings["userId"].nunique()),
        "train_movies_count": int(train_ratings["movieId"].nunique()),
        "outputs": {
            "movie_rating_stats_train": str(MOVIE_RATING_STATS_OUTPUT),
            "user_profiles_train": str(USER_PROFILES_OUTPUT),
            "user_seen_movies_train": str(USER_SEEN_MOVIES_OUTPUT),
        },
    }

    summary_path = OUTPUT_DIR / "eval_train_artifacts_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
