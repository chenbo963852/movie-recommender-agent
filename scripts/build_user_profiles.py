import sys
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


RATINGS_PATH = PROJECT_ROOT / "data/raw/ratings.csv"
MOVIE_DOCS_PATH = PROJECT_ROOT / "data/processed/movie_documents.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/user_profiles.parquet"

CHUNK_SIZE = 1_000_000
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

            rows.append({
                "movieId": int(movie_id),
                "genre": genres
            })

    df = pd.DataFrame(rows)
    df = df.explode("genre")
    df = df.dropna(subset=["genre"])
    df["genre"] = df["genre"].astype(str)

    print(f"Loaded genres for {df['movieId'].nunique()} movies.")
    return df


def build_user_profiles():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    movie_genres_df = load_movie_genres()

    user_stat_parts = []
    liked_stat_parts = []
    genre_pref_parts = []

    print(f"Reading ratings from: {RATINGS_PATH}")

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            RATINGS_PATH,
            usecols=["userId", "movieId", "rating"],
            chunksize=CHUNK_SIZE,
        )
    ):
        user_stats = (
            chunk.groupby("userId")
            .agg(
                rating_count=("rating", "count"),
                rating_sum=("rating", "sum"),
            )
            .reset_index()
        )
        user_stat_parts.append(user_stats)

        liked = chunk[chunk["rating"] >= MIN_LIKED_RATING].copy()

        liked_stats = (
            liked.groupby("userId")
            .agg(
                liked_count=("rating", "count"),
            )
            .reset_index()
        )
        liked_stat_parts.append(liked_stats)

        liked_with_genres = liked.merge(
            movie_genres_df,
            on="movieId",
            how="inner"
        )

        genre_prefs = (
            liked_with_genres.groupby(["userId", "genre"])
            .size()
            .reset_index(name="genre_count")
        )
        genre_pref_parts.append(genre_prefs)

        print(f"Processed chunk {chunk_idx + 1}, rows={len(chunk)}")

    user_stats_all = pd.concat(user_stat_parts, ignore_index=True)
    user_stats_all = (
        user_stats_all.groupby("userId")
        .agg(
            rating_count=("rating_count", "sum"),
            rating_sum=("rating_sum", "sum"),
        )
        .reset_index()
    )

    user_stats_all["avg_rating"] = (
        user_stats_all["rating_sum"] / user_stats_all["rating_count"]
    )

    liked_stats_all = pd.concat(liked_stat_parts, ignore_index=True)
    liked_stats_all = (
        liked_stats_all.groupby("userId")
        .agg(
            liked_count=("liked_count", "sum"),
        )
        .reset_index()
    )

    genre_prefs_all = pd.concat(genre_pref_parts, ignore_index=True)
    genre_prefs_all = (
        genre_prefs_all.groupby(["userId", "genre"])
        .agg(
            genre_count=("genre_count", "sum"),
        )
        .reset_index()
    )

    genre_prefs_all = genre_prefs_all.sort_values(
        ["userId", "genre_count"],
        ascending=[True, False]
    )

    top_genres = genre_prefs_all.groupby("userId").head(5)

    preferred_genres = (
        top_genres.groupby("userId")["genre"]
        .apply(lambda x: ",".join(x.tolist()))
        .reset_index(name="preferred_genres")
    )

    profiles = user_stats_all.merge(
        liked_stats_all,
        on="userId",
        how="left"
    )

    profiles = profiles.merge(
        preferred_genres,
        on="userId",
        how="left"
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

    profiles.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved user profiles to: {OUTPUT_PATH}")
    print(f"Users count: {len(profiles)}")
    print(profiles.head(10))


if __name__ == "__main__":
    build_user_profiles()
