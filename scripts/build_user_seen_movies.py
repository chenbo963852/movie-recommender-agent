import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


RATINGS_PATH = PROJECT_ROOT / "data/raw/ratings.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/user_seen_movies.parquet"

CHUNK_SIZE = 1_000_000


def build_user_seen_movies():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    parts = []

    print(f"Reading ratings from: {RATINGS_PATH}")

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            RATINGS_PATH,
            usecols=["userId", "movieId"],
            chunksize=CHUNK_SIZE,
        )
    ):
        chunk = chunk.drop_duplicates()

        parts.append(chunk)

        print(f"Processed chunk {chunk_idx + 1}, rows={len(chunk)}")

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates()

    user_seen = (
        df.groupby("userId")["movieId"]
        .apply(lambda x: list(map(int, x.tolist())))
        .reset_index(name="seen_movie_ids")
    )

    user_seen.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved user seen movies to: {OUTPUT_PATH}")
    print(f"Users count: {len(user_seen)}")
    print(user_seen.head(10))


if __name__ == "__main__":
    build_user_seen_movies()
