import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


RAW_RATINGS_PATH = PROJECT_ROOT / "data/raw/ratings.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/movie_rating_stats.parquet"

CHUNK_SIZE = 1_000_000


def build_movie_rating_stats():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    partial_results = []

    print(f"Reading ratings from: {RAW_RATINGS_PATH}")

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            RAW_RATINGS_PATH,
            usecols=["movieId", "rating"],
            chunksize=CHUNK_SIZE,
        )
    ):
        grouped = (
            chunk.groupby("movieId")
            .agg(
                rating_count=("rating", "count"),
                rating_sum=("rating", "sum"),
            )
            .reset_index()
        )

        partial_results.append(grouped)

        print(f"Processed chunk {chunk_idx + 1}, rows={len(chunk)}")

    all_stats = pd.concat(partial_results, ignore_index=True)

    final_stats = (
        all_stats.groupby("movieId")
        .agg(
            rating_count=("rating_count", "sum"),
            rating_sum=("rating_sum", "sum"),
        )
        .reset_index()
    )

    final_stats["rating_mean"] = (
        final_stats["rating_sum"] / final_stats["rating_count"]
    )

    final_stats = final_stats[
        ["movieId", "rating_count", "rating_mean"]
    ].sort_values("rating_count", ascending=False)

    final_stats.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved movie rating stats to: {OUTPUT_PATH}")
    print(f"Movies count: {len(final_stats)}")
    print(final_stats.head(10))


if __name__ == "__main__":
    build_movie_rating_stats()
