from pathlib import Path

import pandas as pd


class MovieRatingStatsService:
    def __init__(self):
        self.stats_by_movie_id: dict[int, dict] = {}

    def load_stats(self, path: str = "data/processed/movie_rating_stats.parquet"):
        stats_path = Path(path)

        if not stats_path.exists():
            print(f"Movie rating stats file not found: {stats_path}")
            return

        df = pd.read_parquet(stats_path)

        self.stats_by_movie_id = {
            int(row["movieId"]): {
                "ml_rating_count": int(row["rating_count"]),
                "ml_rating_mean": float(row["rating_mean"]),
            }
            for _, row in df.iterrows()
        }

        print(f"MovieLens rating stats loaded for {len(self.stats_by_movie_id)} movies.")

    def get_stats(self, movie_id: int | None) -> dict:
        if movie_id is None:
            return {
                "ml_rating_count": 0,
                "ml_rating_mean": None,
            }

        try:
            movie_id = int(movie_id)
        except Exception:
            return {
                "ml_rating_count": 0,
                "ml_rating_mean": None,
            }

        return self.stats_by_movie_id.get(
            movie_id,
            {
                "ml_rating_count": 0,
                "ml_rating_mean": None,
            },
        )
