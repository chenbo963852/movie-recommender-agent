import json
from fastapi import FastAPI, Query, HTTPException
from app.services.bm25_service import BM25Service
import time
import logging
import re




from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.schemas import (
    DocumentRequest,
    DocumentBatchRequest,
    StructuredRecommendRequest,
    AgentRecommendRequest,
    AgentUserRecommendRequest,
    LocalAgentUserRecommendRequest,
)
from app.services.ratings_service import RatingsService
from app.evaluation import evaluate_recommendation, evaluate_batch_users
from app.services.retrieval_service import RetrievalService
from app.services.movie_rating_stats_service import MovieRatingStatsService
from app.services.user_profile_service import UserProfileService
from app.services.user_seen_movies_service import UserSeenMoviesService
from app.services.intent_parser_service import IntentParserService
from app.services.agent_recommendation_service import AgentRecommendationService
from app.services.local_llm_service import LocalLLMService
from pydantic import BaseModel
from app.services.recommendation_service import UserRecommendationService











app = FastAPI()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


embedding_service = EmbeddingService()
qdrant_service = QdrantService()
bm25_service = BM25Service()
ratings_service = RatingsService()
movie_rating_stats_service = MovieRatingStatsService()
user_profile_service = UserProfileService()
user_seen_movies_service = UserSeenMoviesService()
intent_parser_service = IntentParserService()
local_llm_service = LocalLLMService()




retrieval_service = RetrievalService(
    embedding_service=embedding_service,
    qdrant_service=qdrant_service,
    bm25_service=bm25_service,
    movie_rating_stats_service=movie_rating_stats_service,
)

agent_recommendation_service = AgentRecommendationService(
    intent_parser_service=intent_parser_service,
    retrieval_service=retrieval_service,
    user_profile_service=user_profile_service,
    user_seen_movies_service=user_seen_movies_service,
)

user_recommendation_service = UserRecommendationService(
    ratings_service=ratings_service,
    qdrant_service=qdrant_service,
    embedding_service=embedding_service,
    retrieval_service=retrieval_service,
)



class LocalAgentRecommendRequest(BaseModel):
    prompt: str



def init_data():
    with open("data/documents.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    qdrant_service.recreate_collection(vector_size=768)

    texts = [doc["text"] for doc in docs]
    vectors = embedding_service.encode_batch(texts)
    qdrant_service.add_documents(docs, vectors)

    bm25_service.build_index(docs)

def load_documents_from_json():
        with open("data/documents.json", "r", encoding="utf-8") as f:
            return json.load(f)

def save_documents_to_json(docs: list[dict]):
        with open("data/documents.json", "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)

def document_id_exists(doc_id: int, docs: list[dict]) -> bool:
    for doc in docs:
        if doc["id"] == doc_id:
            return True
    return False

def rebuild_bm25_index():
    docs = load_documents_from_json()
    bm25_service.build_index(docs, force_rebuild=True)
    print(f"BM25 index rebuilt with {len(docs)} documents.")


def extract_year(title: str) -> int | None:
    match = re.search(r"\((\d{4})\)", title)
    if match:
        return int(match.group(1))
    return None

def get_user_seed_movies(
    user_id: int,
    min_rating: float = 4.0,
    max_seeds: int = 3,
    excluded_movie_ids: set[int] | None = None
):
    ratings = ratings_service.get_user_ratings(user_id)

    high_rated = [
        item for item in ratings
        if item["rating"] >= min_rating
        and (excluded_movie_ids is None or item["movie_id"] not in excluded_movie_ids)
    ]
    high_rated.sort(key=lambda x: x["rating"], reverse=True)

    return high_rated[:max_seeds]


def get_user_watched_movie_ids(
    user_id: int,
    excluded_movie_ids: set[int] | None = None
):
    ratings = ratings_service.get_user_ratings(user_id)
    watched = {item["movie_id"] for item in ratings}

    if excluded_movie_ids:
        watched -= excluded_movie_ids

    return watched

def recommend_for_user_core(
    user_id: int,
    top_k: int = 20,
    min_rating: float = 4.0,
    seed_limit: int = 10,
    excluded_movie_ids: set[int] | None = None,
    exclude_watched: bool = True
):
    if not ratings_service.is_loaded():
        raise ValueError("Ratings data not loaded")

    user_ratings = ratings_service.get_user_ratings(user_id)
    if not user_ratings:
        raise ValueError(f"User id {user_id} not found")

    seed_movies = get_user_seed_movies(
        user_id=user_id,
        min_rating=min_rating,
        max_seeds=seed_limit,
        excluded_movie_ids=excluded_movie_ids
    )

    if not seed_movies:
        return {
            "user_id": user_id,
            "seed_movies": [],
            "results": []
        }

    seed_movie_details = []
    for seed in seed_movies:
        seed_movie = qdrant_service.get_document_by_id(seed["movie_id"])
        if seed_movie is not None:
            seed_movie_details.append({
                "movie_id": seed["movie_id"],
                "title": seed_movie.payload.get("text"),
                "genres": seed_movie.payload.get("category"),
                "rating": seed["rating"]
            })

    watched_movie_ids = get_user_watched_movie_ids(
        user_id=user_id,
        excluded_movie_ids=excluded_movie_ids
    )

    candidate_map = {}

    for seed in seed_movies:
        seed_movie_id = seed["movie_id"]
        seed_rating = seed["rating"]

        target_movie = qdrant_service.get_document_by_id(seed_movie_id)
        if target_movie is None:
            continue

        target_text = target_movie.payload.get("text", "")
        target_category = target_movie.payload.get("category", "")

        seed_title = target_text
        seed_genres = target_category

        target_genres = set(target_category.lower().split())
        target_year = extract_year(target_text)

        query_vector = embedding_service.encode(target_text)
        candidates = qdrant_service.search(query_vector, top_k=300)

        for point in candidates:
            candidate_id = point.id

            # 屏蔽种子自身
            if candidate_id == seed_movie_id:
                continue
            # 屏蔽自定义排除的电影 + 已看电影
            if exclude_watched:
                if (excluded_movie_ids and candidate_id in excluded_movie_ids) or candidate_id in watched_movie_ids:
                    continue

            candidate_title = point.payload.get("text", "")
            candidate_genres_text = point.payload.get("category", "")
            candidate_genres = set(candidate_genres_text.lower().split())

            shared_genres = target_genres & candidate_genres
            genre_overlap_count = len(shared_genres)
            genre_penalty = 1.0 if genre_overlap_count > 0 else 0.2

            vector_score = float(point.score)

            candidate_year = extract_year(candidate_title)
            year_bonus = 0.0
            year_gap = None

            if target_year is not None and candidate_year is not None:
                year_gap = abs(target_year - candidate_year)
                if year_gap <= 3:
                    year_bonus = 1.0
                elif year_gap <= 10:
                    year_bonus = 0.5
                elif year_gap <= 20:
                    year_bonus = 0.2

            # 优化后打分公式
            final_score = (
                0.45 * vector_score
                + 0.35 * genre_overlap_count
                + 0.15 * year_bonus
                + 0.05 * seed_rating
                + 0.05 * (1.0 if seed_rating >= 4.5 else 0.0)
            ) * genre_penalty

            if candidate_id not in candidate_map or final_score > candidate_map[candidate_id]["final_score"]:
                candidate_map[candidate_id] = {
                    "id": candidate_id,
                    "title": candidate_title,
                    "genres": candidate_genres_text,
                    "vector_score": vector_score,
                    "genre_overlap_count": genre_overlap_count,
                    "shared_genres": sorted(shared_genres),
                    "candidate_year": candidate_year,
                    "year_gap": year_gap,
                    "year_bonus": year_bonus,
                    "source_seed_movie_id": seed_movie_id,
                    "source_seed_title": seed_title,
                    "source_seed_genres": seed_genres,
                    "source_seed_rating": seed_rating,
                    "final_score": float(final_score)
                }

    results = list(candidate_map.values())
    results.sort(key=lambda x: x["final_score"], reverse=True)
    final_results = results[:top_k]

    return {
        "user_id": user_id,
        "seed_movies": seed_movie_details,
        "results": final_results
    }






@app.on_event("startup")
def startup():
    if not bm25_service.load_cache_if_exists():
        rebuild_bm25_index()
        print("BM25 index rebuilt from documents.json.")
    else:
        print("BM25 index loaded from cache.")

    movie_rating_stats_service.load_stats()
    user_profile_service.load_profiles()
    user_seen_movies_service.load_seen_movies()

    ratings_service.load_ratings()
    print(f"Ratings loaded for {len(ratings_service.movie_stats)} movies.")

    local_llm_service.load_model()




@app.get("/")
def root():
    return {"message": "Qdrant local demo is running"}



def search_movies_core(
    query: str,
    top_k: int = 5,
    category: str | None = None,
    genre: str | None = None,
    exclude_genre: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    min_vote_average: float | None = None,
    min_vote_count: int | None = None
):
    return retrieval_service.search_movies(
        query=query,
        top_k=top_k,
        category=category,
        genre=genre,
        exclude_genre=exclude_genre,
        year_from=year_from,
        year_to=year_to,
        min_vote_average=min_vote_average,
        min_vote_count=min_vote_count,
    )



@app.get("/search")
def search(
    query: str,
    top_k: int = Query(default=5, ge=1, le=20),
    category: str | None = None,
    genre: str | None = None,
    exclude_genre: str | None = None,
    year_from: int | None = Query(default=None, ge=1800, le=2030),
    year_to: int | None = Query(default=None, ge=1800, le=2030),
    min_vote_average: float | None = Query(default=None, ge=0.0, le=10.0),
    min_vote_count: int | None = None
):
    return search_movies_core(
        query=query,
        top_k=top_k,
        category=category,
        genre=genre,
        exclude_genre=exclude_genre,
        year_from=year_from,
        year_to=year_to,
        min_vote_average=min_vote_average,
        min_vote_count=min_vote_count
    )


@app.post("/recommend/structured")
def recommend_structured(req: StructuredRecommendRequest):
    return search_movies_core(
        query=req.query,
        top_k=req.top_k,
        genre=req.genre,
        exclude_genre=req.exclude_genre,
        year_from=req.year_from,
        year_to=req.year_to,
        min_vote_average=req.min_vote_average,
        min_vote_count=req.min_vote_count
    )






@app.post("/documents")
def add_document(doc: DocumentRequest):
    docs = load_documents_from_json()

    if document_id_exists(doc.id, docs):
        raise HTTPException(
            status_code=400,
            detail=f"Document id {doc.id} already exists"
        )

    vector = embedding_service.encode(doc.text)

    # 1. 写入 Qdrant
    qdrant_service.add_document(
        doc={
            "id": doc.id,
            "text": doc.text,
            "category": doc.category
        },
        vector=vector
    )

    # 2. 写入 documents.json
    new_doc = {
        "id": doc.id,
        "text": doc.text,
        "category": doc.category
    }
    docs.append(new_doc)
    save_documents_to_json(docs)

    # 3. 重建 BM25
    rebuild_bm25_index()

    return {
        "message": "document added successfully",
        "document": {
            "id": new_doc["id"],
            "title": new_doc["text"],
            "genres": new_doc["category"]
        }
    }


@app.post("/documents/batch")
def add_documents_batch(batch: DocumentBatchRequest):
    current_docs = load_documents_from_json()
    current_ids = {doc["id"] for doc in current_docs}

    new_ids = [doc.id for doc in batch.documents]

    # 检查和已有数据是否重复
    duplicate_existing = [doc_id for doc_id in new_ids if doc_id in current_ids]

    # 检查这一批内部是否重复
    duplicate_inside_batch = []
    seen = set()
    for doc_id in new_ids:
        if doc_id in seen and doc_id not in duplicate_inside_batch:
            duplicate_inside_batch.append(doc_id)
        seen.add(doc_id)

    if duplicate_existing or duplicate_inside_batch:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Duplicate document ids found",
                "already_exists": duplicate_existing,
                "duplicated_in_batch": duplicate_inside_batch
            }
        )

    texts = [doc.text for doc in batch.documents]
    vectors = embedding_service.encode_batch(texts)

    docs_to_add = [
        {
            "id": doc.id,
            "text": doc.text,
            "category": doc.category
        }
        for doc in batch.documents
    ]

    # 1. 写入 Qdrant
    qdrant_service.add_documents(
        docs=docs_to_add,
        vectors=vectors
    )

    # 2. 写入 documents.json
    current_docs.extend(docs_to_add)
    save_documents_to_json(current_docs)

    # 3. 重建 BM25
    rebuild_bm25_index()

    return {
        "message": "batch insert success",
        "count": len(docs_to_add)
    }



@app.get("/documents")
def get_documents(limit: int = Query(default=10, ge=1, le=100)):
    points = qdrant_service.get_documents(limit=limit)

    documents = []
    for point in points:
        documents.append(
            {
                "id": point.id,
                "title": point.payload.get("text"),
                "genres": point.payload.get("category")
            }
        )

    return {
        "count": len(documents),
        "documents": documents
    }


@app.get("/recommend/similar")
def recommend_similar(
    movie_id: int,
    top_k: int = Query(default=5, ge=1, le=20),
    category: str | None = None
):
    start_time = time.time()

    # 1. 取目标电影
    target_movie = qdrant_service.get_document_by_id(movie_id)

    if target_movie is None:
        raise HTTPException(
            status_code=404,
            detail=f"Movie id {movie_id} not found"
        )

    target_text = target_movie.payload.get("text", "")
    target_category = target_movie.payload.get("category", "")

    # 把目标电影 genres 拆成集合
    target_genres = set(target_category.lower().split())
    target_year = extract_year(target_text)

    # 2. 用目标电影标题做向量检索
    query_vector = embedding_service.encode(target_text)

    candidates = qdrant_service.search(query_vector, top_k=50)

    # 3. 去掉自己
    filtered = [point for point in candidates if point.id != movie_id]

    # 4. 如果传了 category，先做包含匹配过滤
    if category:
        category_lower = category.lower()
        filtered = [
            point for point in filtered
            if category_lower in point.payload.get("category", "").lower()
        ]

    # 5. genre overlap 重排
    reranked = []
    for point in filtered:
        candidate_text = point.payload.get("text", "")
        candidate_category = point.payload.get("category", "")
        candidate_genres = set(candidate_category.lower().split())

        shared_genres = target_genres & candidate_genres
        genre_overlap_count = len(shared_genres)

        vector_score = float(point.score)

        candidate_year = extract_year(candidate_text)

        year_bonus = 0.0
        year_gap = None

        if target_year is not None and candidate_year is not None:
            year_gap = abs(target_year - candidate_year)

            if year_gap <= 3:
                year_bonus = 1.0
            elif year_gap <= 10:
                year_bonus = 0.5
            elif year_gap <= 20:
                year_bonus = 0.2
            else:
                year_bonus = 0.0

        final_score = (
                0.6 * vector_score
                + 0.25 * genre_overlap_count
                + 0.15 * year_bonus
        )

        reranked.append(
            {
                "id": point.id,
                "title": candidate_text,
                "genres": candidate_category,
                "vector_score": vector_score,
                "genre_overlap_count": genre_overlap_count,
                "shared_genres": sorted(shared_genres),
                "candidate_year": candidate_year,
                "year_gap": year_gap,
                "year_bonus": year_bonus,
                "final_score": float(final_score)
            }
        )

    # 6. 按最终分数排序
    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    final_results = reranked[:top_k]

    elapsed_time = time.time() - start_time

    logger.info(
        f"Recommend similar movie_id={movie_id}, title='{target_text}', "
        f"category_filter='{category}', top_k={top_k}, "
        f"result_count={len(final_results)}, elapsed_time={elapsed_time:.4f}s"
    )

    return {
        "movie_id": movie_id,
        "movie_title": target_text,
        "movie_genres": target_category,
        "results": final_results,
        "elapsed_time_seconds": round(elapsed_time, 4)
    }




@app.get("/movies/popular")
def get_popular_movies(
    top_k: int = Query(default=10, ge=1, le=50),
    genre: str | None = None,
    min_rating_count: int = Query(default=20, ge=1)
):
    start_time = time.time()

    if not ratings_service.is_loaded():
        raise HTTPException(
            status_code=500,
            detail="Ratings data not loaded"
        )

    all_points = qdrant_service.get_all_documents(batch_size=500)

    results = []

    for point in all_points:
        movie_id = point.id
        title = point.payload.get("text")
        genres = point.payload.get("category", "")

        stats = ratings_service.get_movie_stats(movie_id)
        if not stats:
            continue

        if stats["rating_count"] < min_rating_count:
            continue

        if genre:
            if genre.lower() not in genres.lower():
                continue

        results.append(
            {
                "id": movie_id,
                "title": title,
                "genres": genres,
                "avg_rating": stats["avg_rating"],
                "rating_count": stats["rating_count"],
                "popular_score": stats["popular_score"]
            }
        )

    results.sort(key=lambda x: x["popular_score"], reverse=True)
    final_results = results[:top_k]

    elapsed_time = time.time() - start_time

    logger.info(
        f"Popular movies requested, genre='{genre}', top_k={top_k}, "
        f"min_rating_count={min_rating_count}, result_count={len(final_results)}, "
        f"elapsed_time={elapsed_time:.4f}s"
    )

    return {
        "genre": genre,
        "top_k": top_k,
        "min_rating_count": min_rating_count,
        "results": final_results,
        "elapsed_time_seconds": round(elapsed_time, 4)
    }


@app.get("/user/recommend")
def user_recommend(
    user_id: int,
    top_k: int = Query(default=5, ge=1, le=20),
):
    profile = user_profile_service.get_profile(user_id)

    if profile is None:
        return {
            "user_id": user_id,
            "message": "User profile not found.",
            "results": []
        }

    seen_movie_ids = user_seen_movies_service.get_seen_movie_ids(user_id)

    preferred_genres_text = profile.get("preferred_genres") or ""

    preferred_genres = [
        genre.strip()
        for genre in preferred_genres_text.split(",")
        if genre.strip()
    ]

    if preferred_genres:
        query = (
            "highly rated hidden gem movies for a user who likes "
            + ", ".join(preferred_genres)
        )
    else:
        query = "highly rated hidden gem movies"

    search_genres = preferred_genres[:3] if preferred_genres else [None]

    merged = {}

    def add_candidates(
        search_result: dict,
        matched_genre: str | None,
        genre_rank: int,
        is_fallback: bool = False,
    ):
        for movie in search_result["results"]:
            movie_id = movie["id"]

            if movie_id in seen_movie_ids:
                continue

            genre_boost = max(0.15 - genre_rank * 0.05, 0.0)

            fallback_penalty = 0.03 if is_fallback else 0.0

            personalization_score = (
                movie["final_score"]
                + genre_boost
                - fallback_penalty
            )

            if movie_id not in merged:
                movie["matched_preferred_genre"] = matched_genre
                movie["genre_boost"] = round(genre_boost, 4)
                movie["fallback_candidate"] = is_fallback
                movie["personalization_score"] = round(personalization_score, 4)

                reason_parts = []

                if matched_genre:
                    reason_parts.append(f"matches user's preferred genre: {matched_genre}")

                if movie.get("ml_rating_mean") is not None:
                    reason_parts.append(
                        f"MovieLens mean rating {round(movie['ml_rating_mean'], 2)} "
                        f"from {movie.get('ml_rating_count', 0)} ratings"
                    )

                if movie.get("tmdb_quality_score") is not None:
                    reason_parts.append(
                        f"TMDB quality score {movie['tmdb_quality_score']}"
                    )

                if is_fallback:
                    reason_parts.append("selected from fallback recall because strict candidates were limited")

                movie["recommendation_reason"] = "; ".join(reason_parts)

                merged[movie_id] = movie
            else:
                if personalization_score > merged[movie_id]["personalization_score"]:
                    merged[movie_id]["matched_preferred_genre"] = matched_genre
                    merged[movie_id]["genre_boost"] = round(genre_boost, 4)
                    merged[movie_id]["fallback_candidate"] = is_fallback
                    merged[movie_id]["personalization_score"] = round(personalization_score, 4)

                    reason_parts = []

                    if matched_genre:
                        reason_parts.append(f"matches user's preferred genre: {matched_genre}")

                    if movie.get("ml_rating_mean") is not None:
                        reason_parts.append(
                            f"MovieLens mean rating {round(movie['ml_rating_mean'], 2)} "
                            f"from {movie.get('ml_rating_count', 0)} ratings"
                        )

                    if movie.get("tmdb_quality_score") is not None:
                        reason_parts.append(
                            f"TMDB quality score {movie['tmdb_quality_score']}"
                        )

                    if is_fallback:
                        reason_parts.append("selected from fallback recall because strict candidates were limited")

                    merged[movie_id]["recommendation_reason"] = "; ".join(reason_parts)

    # 第一轮：按用户 top 3 genre 精准召回
    for genre_rank, genre in enumerate(search_genres):
        search_result = retrieval_service.search_movies(
            query=query,
            top_k=200,
            genre=genre,
            min_vote_average=6.5,
            min_vote_count=300,
        )

        add_candidates(
            search_result=search_result,
            matched_genre=genre,
            genre_rank=genre_rank,
            is_fallback=False,
        )

    # 第二轮：如果候选不够，放宽质量门槛兜底
    if len(merged) < top_k:
        for genre_rank, genre in enumerate(search_genres):
            search_result = retrieval_service.search_movies(
                query=query,
                top_k=200,
                genre=genre,
                min_vote_average=6.0,
                min_vote_count=50,
            )

            add_candidates(
                search_result=search_result,
                matched_genre=genre,
                genre_rank=genre_rank,
                is_fallback=True,
            )

    # 第三轮：如果还不够，不限制 genre 兜底
    if len(merged) < top_k:
        search_result = retrieval_service.search_movies(
            query=query,
            top_k=200,
            genre=None,
            min_vote_average=6.0,
            min_vote_count=50,
        )

        add_candidates(
            search_result=search_result,
            matched_genre=None,
            genre_rank=3,
            is_fallback=True,
        )

    results = list(merged.values())
    results.sort(key=lambda x: x["personalization_score"], reverse=True)
    final_results = results[:top_k]

    return {
        "user_id": user_id,
        "profile": profile,
        "generated_query": query,
        "used_preferred_genres": search_genres,
        "exclude_seen": True,
        "seen_movie_count": len(seen_movie_ids),
        "candidate_count_after_excluding_seen": len(results),
        "results": final_results,
    }

@app.post("/agent/recommend")
def agent_recommend(req: AgentRecommendRequest):
    return agent_recommendation_service.recommend(req.message)

@app.post("/agent/user-recommend")
def agent_user_recommend(req: AgentUserRecommendRequest):
    return agent_recommendation_service.recommend_for_user(
        user_id=req.user_id,
        message=req.message,
    )

@app.get("/llm/test")
def test_local_llm(prompt: str):
    response = local_llm_service.chat(prompt)

    return {
        "prompt": prompt,
        "response": response
    }

@app.post("/agent/recommend/local")
def local_agent_recommend(req: LocalAgentRecommendRequest):
    params = local_llm_service.extract_search_params(req.prompt)

    return agent_recommendation_service.recommend_with_parsed_intent(
        message=req.prompt,
        parsed_intent=params,
        agent_type="local_llm_agent",
    )

@app.post("/agent/recommend/local/user")
def local_agent_user_recommend(req: LocalAgentUserRecommendRequest):
    params = local_llm_service.extract_search_params(req.prompt)

    return agent_recommendation_service.recommend_for_user_with_parsed_intent(
        user_id=req.user_id,
        message=req.prompt,
        parsed_intent=params,
        agent_type="local_llm_personalized_agent",
    )



# -------------------------- 评估接口 --------------------------
@app.get("/user/recommend/evaluate")
def evaluate_user_recommend(
    user_id: int,
    top_k: int = Query(10, ge=10, le=100),
    seed_limit: int = Query(3, ge=1, le=20),
    min_rating: float = Query(4.0, ge=0.5, le=5.0)
):
    try:
        return evaluate_recommendation(
            ratings_service=ratings_service,
            qdrant_service=qdrant_service,
            recommend_for_user_core=user_recommendation_service.recommend_for_user,
            user_id=user_id,
            top_k=top_k,
            seed_limit=seed_limit,
            min_rating=min_rating
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/user/recommend/evaluate/batch")
def evaluate_batch_recommend(
    user_ids: str = Query(description="逗号分隔，如 1,2,3"),
    top_k: int = Query(10, ge=10, le=100),
    min_rating: float = Query(4.0, ge=0.5, le=5.0),
    seed_limit: int = Query(3, ge=1, le=20)
):
    try:
        uid_list = [int(x.strip()) for x in user_ids.split(",")]
    except:
        raise HTTPException(status_code=400, detail="用户ID格式错误")

    return evaluate_batch_users(
        ratings_service=ratings_service,
        qdrant_service=qdrant_service,
        recommend_for_user_core=user_recommendation_service.recommend_for_user,
        user_ids=uid_list,
        top_k=top_k,
        min_rating=min_rating,
        seed_limit=seed_limit
    )





