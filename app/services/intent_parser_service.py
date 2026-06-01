import re


class IntentParserService:
    def parse_recommend_intent(self, message: str) -> dict:
        text = message.lower()

        genre_map = {
            "科幻": "Science Fiction",
            "science fiction": "Science Fiction",
            "sci-fi": "Science Fiction",

            "惊悚": "Thriller",
            "thriller": "Thriller",

            "剧情": "Drama",
            "drama": "Drama",

            "喜剧": "Comedy",
            "comedy": "Comedy",

            "爱情": "Romance",
            "romance": "Romance",

            "动作": "Action",
            "action": "Action",

            "犯罪": "Crime",
            "crime": "Crime",

            "纪录片": "Documentary",
            "documentary": "Documentary",

            "动画": "Animation",
            "animation": "Animation",

            "恐怖": "Horror",
            "horror": "Horror",
        }

        genres = []
        exclude_genres = []

        for keyword, genre in genre_map.items():
            if keyword not in text:
                continue

            is_excluded = (
                f"不要{keyword}" in text
                or f"不想看{keyword}" in text
                or f"排除{keyword}" in text
                or f"不要 {keyword}" in text
                or f"no {keyword}" in text
                or f"exclude {keyword}" in text
                or f"without {keyword}" in text
            )

            if is_excluded:
                if genre not in exclude_genres:
                    exclude_genres.append(genre)
            else:
                if genre not in genres:
                    genres.append(genre)

        year_from = None
        year_to = None

        match = re.search(r"(\d{4})年之后", message)
        if match:
            year_from = int(match.group(1))

        match = re.search(r"(\d{4})年以后", message)
        if match:
            year_from = int(match.group(1))

        match = re.search(r"after\s+(\d{4})", text)
        if match:
            year_from = int(match.group(1))

        match = re.search(r"(\d{4})年之前", message)
        if match:
            year_to = int(match.group(1))

        match = re.search(r"(\d{4})年以前", message)
        if match:
            year_to = int(match.group(1))

        match = re.search(r"before\s+(\d{4})", text)
        if match:
            year_to = int(match.group(1))

        min_vote_average = None

        if "高评分" in message or "好评" in message or "highly rated" in text:
            min_vote_average = 7.0
        elif "中等评分" in message or "一般评分" in message or "medium rated" in text:
            min_vote_average = 5.0
        elif "低评分" in message or "随便" in message or "low rated" in text:
            min_vote_average = 1.0

        min_vote_count = None

        match = re.search(r"至少\s*(\d+)\s*人", message)
        if match:
            min_vote_count = int(match.group(1))

        match = re.search(r"至少\s*(\d+)\s*个评分", message)
        if match:
            min_vote_count = int(match.group(1))

        match = re.search(r"at least\s*(\d+)\s*ratings", text)
        if match:
            min_vote_count = int(match.group(1))

        top_k = 5

        match = re.search(r"推荐\s*(\d+)\s*部", message)
        if match:
            top_k = int(match.group(1))

        match = re.search(r"(\d+)\s*部", message)
        if match:
            top_k = int(match.group(1))

        query = self.build_english_query(
            genres=genres,
            exclude_genres=exclude_genres,
            year_from=year_from,
            year_to=year_to,
            min_vote_average=min_vote_average,
            original_message=message,
        )

        return {
            "query": query,
            "genre": ",".join(genres) if genres else None,
            "exclude_genre": ",".join(exclude_genres) if exclude_genres else None,
            "year_from": year_from,
            "year_to": year_to,
            "min_vote_average": min_vote_average,
            "min_vote_count": min_vote_count,
            "top_k": top_k,
        }

    def build_english_query(
        self,
        genres: list[str],
        exclude_genres: list[str],
        year_from: int | None,
        year_to: int | None,
        min_vote_average: float | None,
        original_message: str,
    ) -> str:
        parts = []

        if min_vote_average is not None and min_vote_average >= 7.0:
            parts.append("highly rated")
        elif min_vote_average is not None and min_vote_average >= 5.0:
            parts.append("medium rated")
        else:
            parts.append("recommended")

        if genres:
            genre_text = " ".join(genres).lower()
            parts.append(genre_text)

        parts.append("movies")

        if year_from is not None:
            parts.append(f"after {year_from}")

        if year_to is not None:
            parts.append(f"before {year_to}")

        if exclude_genres:
            excluded_text = " ".join(exclude_genres).lower()
            parts.append(f"excluding {excluded_text}")

        return " ".join(parts)
