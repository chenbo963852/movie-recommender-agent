import ast
import json
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

MOVIES_FILE = RAW_DIR / "movies_metadata.csv"
CREDITS_FILE = RAW_DIR / "credits.csv"
KEYWORDS_FILE = RAW_DIR / "keywords.csv"
LINKS_FILE = RAW_DIR / "links.csv"

OUTPUT_FILE = PROCESSED_DIR / "movie_documents.jsonl"


def safe_parse_list(value):
    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return []

    return []


def extract_names(value, limit=None):
    items = safe_parse_list(value)
    names = []

    for item in items:
        name = item.get("name")
        if name:
            names.append(name)

    if limit:
        names = names[:limit]

    return names


def extract_directors(crew_value):
    items = safe_parse_list(crew_value)
    directors = []

    for item in items:
        if item.get("job") == "Director" and item.get("name"):
            directors.append(item["name"])

    return directors


def extract_year(date_value):
    if pd.isna(date_value):
        return None

    text = str(date_value)
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])

    return None


def build_text(row):
    parts = [
        f"Title: {row['title']}",
        f"Genres: {', '.join(row['genres'])}",
        f"Overview: {row['overview']}",
        f"Keywords: {', '.join(row['keywords'])}",
        f"Cast: {', '.join(row['cast'])}",
        f"Directors: {', '.join(row['directors'])}",
        f"Year: {row['year']}",
    ]

    return "\n".join([p for p in parts if p and not p.endswith(": ")])


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading movies_metadata.csv")
    movies = pd.read_csv(MOVIES_FILE, low_memory=False)

    print("Loading credits.csv")
    credits = pd.read_csv(CREDITS_FILE)

    print("Loading keywords.csv")
    keywords = pd.read_csv(KEYWORDS_FILE)

    print("Loading links.csv")
    links = pd.read_csv(LINKS_FILE)

    print("Cleaning ids")

    movies = movies[movies["id"].astype(str).str.isdigit()].copy()
    movies["tmdb_id"] = movies["id"].astype(int)

    links = links.dropna(subset=["tmdbId"]).copy()
    links["tmdb_id"] = links["tmdbId"].astype(int)
    links["movie_id"] = links["movieId"].astype(int)

    credits["tmdb_id"] = credits["id"].astype(int)
    keywords["tmdb_id"] = keywords["id"].astype(int)

    print("Merging data")

    df = links.merge(movies, on="tmdb_id", how="inner")
    df = df.merge(credits[["tmdb_id", "cast", "crew"]], on="tmdb_id", how="left")
    df = df.merge(keywords[["tmdb_id", "keywords"]], on="tmdb_id", how="left")

    print(f"Merged movies: {len(df)}")

    count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            title = row.get("title")
            overview = row.get("overview")

            if pd.isna(title) or pd.isna(overview):
                continue

            genres = extract_names(row.get("genres"))
            keyword_names = extract_names(row.get("keywords"), limit=20)
            cast_names = extract_names(row.get("cast"), limit=10)
            directors = extract_directors(row.get("crew"))
            year = extract_year(row.get("release_date"))

            doc = {
                "id": int(row["movie_id"]),
                "movie_id": int(row["movie_id"]),
                "tmdb_id": int(row["tmdb_id"]),
                "imdb_id": row.get("imdb_id"),
                "title": title,
                "overview": overview,
                "genres": genres,
                "keywords": keyword_names,
                "cast": cast_names,
                "directors": directors,
                "year": year,
                "runtime": None if pd.isna(row.get("runtime")) else row.get("runtime"),
                "vote_average": None if pd.isna(row.get("vote_average")) else row.get("vote_average"),
                "vote_count": None if pd.isna(row.get("vote_count")) else row.get("vote_count"),
                "popularity": None if pd.isna(row.get("popularity")) else row.get("popularity"),
                "text": "",
                "category": " ".join(genres) if genres else "Unknown",
            }

            doc["text"] = build_text(doc)

            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1

    print(f"Done. Saved {count} movie documents to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
