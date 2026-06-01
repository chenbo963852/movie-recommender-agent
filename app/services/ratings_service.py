import csv
import math


class RatingsService:
    def __init__(self, ratings_file: str = "data/ratings.csv"):
        self.ratings_file = ratings_file
        self.movie_stats = {}
        self.user_ratings = {}

    def load_ratings(self):
        stats = {}
        user_ratings = {}

        with open(self.ratings_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                user_id = int(row["userId"])
                movie_id = int(row["movieId"])
                rating = float(row["rating"])

                if movie_id not in stats:
                    stats[movie_id] = {
                        "sum_rating": 0.0,
                        "count": 0
                    }

                stats[movie_id]["sum_rating"] += rating
                stats[movie_id]["count"] += 1

                if user_id not in user_ratings:
                    user_ratings[user_id] = []

                user_ratings[user_id].append({
                    "movie_id": movie_id,
                    "rating": rating
                })

        self.movie_stats = {}
        for movie_id, item in stats.items():
            avg_rating = item["sum_rating"] / item["count"]
            rating_count = item["count"]
            popular_score = avg_rating * math.log1p(rating_count)

            self.movie_stats[movie_id] = {
                "avg_rating": round(avg_rating, 4),
                "rating_count": rating_count,
                "popular_score": round(popular_score, 4)
            }

        self.user_ratings = user_ratings

    def get_movie_stats(self, movie_id: int):
        return self.movie_stats.get(movie_id)

    def get_user_ratings(self, user_id: int):
        return self.user_ratings.get(user_id, [])

    def is_loaded(self) -> bool:
        return len(self.movie_stats) > 0
