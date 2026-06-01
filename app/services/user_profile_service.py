from pathlib import Path

import pandas as pd


class UserProfileService:
    def __init__(self):
        self.profiles_by_user_id: dict[int, dict] = {}

    def load_profiles(self, path: str = "data/processed/user_profiles.parquet"):
        profiles_path = Path(path)

        if not profiles_path.exists():
            print(f"User profiles file not found: {profiles_path}")
            return

        df = pd.read_parquet(profiles_path)

        self.profiles_by_user_id = {
            int(row["userId"]): {
                "user_id": int(row["userId"]),
                "rating_count": int(row["rating_count"]),
                "liked_count": int(row["liked_count"]),
                "avg_rating": float(row["avg_rating"]),
                "preferred_genres": row["preferred_genres"],
            }
            for _, row in df.iterrows()
        }

        print(f"User profiles loaded for {len(self.profiles_by_user_id)} users.")

    def get_profile(self, user_id: int | None) -> dict | None:
        if user_id is None:
            return None

        try:
            user_id = int(user_id)
        except Exception:
            return None

        return self.profiles_by_user_id.get(user_id)
