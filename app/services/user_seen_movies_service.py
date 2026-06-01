from pathlib import Path

import pandas as pd


class UserSeenMoviesService:
    def __init__(self):
        self.seen_movies_by_user_id: dict[int, set[int]] = {}

    def load_seen_movies(self, path: str = "data/processed/user_seen_movies.parquet"):
        seen_path = Path(path)

        if not seen_path.exists():
            print(f"User seen movies file not found: {seen_path}")
            return

        df = pd.read_parquet(seen_path)

        self.seen_movies_by_user_id = {
            int(row["userId"]): set(map(int, row["seen_movie_ids"]))
            for _, row in df.iterrows()
        }

        print(f"User seen movies loaded for {len(self.seen_movies_by_user_id)} users.")

    def get_seen_movie_ids(self, user_id: int | None) -> set[int]:
        if user_id is None:
            return set()

        try:
            user_id = int(user_id)
        except Exception:
            return set()

        return self.seen_movies_by_user_id.get(user_id, set())
