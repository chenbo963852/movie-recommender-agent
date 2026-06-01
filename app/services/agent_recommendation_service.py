class AgentRecommendationService:
    def __init__(
        self,
        intent_parser_service,  
        retrieval_service, 
        user_profile_service=None,
        user_seen_movies_service=None,
    ):
        self.intent_parser_service = intent_parser_service # 解析自然语言成推荐参数
        self.retrieval_service = retrieval_service # 检索电影，得到候选集
        self.user_profile_service = user_profile_service # 获取用户个人画像
        self.user_seen_movies_service = user_seen_movies_service # 获取用户已看过的电影

    def parse_genre_list(self, value: str | None) -> list[str]:
        if not value:
            return []

        return [
            item.strip()
            for item in value.split(",") # 把按逗号分隔的字符串转成列表
            if item.strip()
        ]

    def get_matched_required_genres( # 把按逗号分隔的字符串转成列表
        self,
        movie_genres: str | None,
        required_genres: list[str],
    ) -> list[str]:
        if not movie_genres or not required_genres:
            return []

        movie_genres_lower = movie_genres.lower()

        return [
            genre
            for genre in required_genres
            if genre.lower() in movie_genres_lower
        ]

    def get_strategy_adjustment(self, strategy: str) -> float:
        if strategy == "strict":
            return 0.08

        if strategy == "relax_quality_filters":
            return 0.06

        if strategy == "relax_multi_genre_to_primary_genre":
            return -0.07

        if strategy == "broad_semantic_fallback":
            return -0.10

        return 0.0

    def build_agent_reason(self, movie: dict, parsed_intent: dict) -> str:
        reason_parts = []

        matched_genres = movie.get("matched_required_genres") or []
        required_match_count = movie.get("required_genre_match_count", 0)
        required_total_count = movie.get("required_genre_total_count", 0)

        if matched_genres:
            if required_total_count > 0 and required_match_count == required_total_count:
                reason_parts.append(
                    "matched all required genres: " + ", ".join(matched_genres)
                )
            else:
                reason_parts.append(
                    "partially matched required genres: " + ", ".join(matched_genres)
                )
        else:
            if required_total_count > 0:
                reason_parts.append(
                    "did not match required genres, used only as low-priority semantic fallback"
                )

        strategy = movie.get("agent_strategy")

        if strategy == "strict":
            reason_parts.append("returned by strict intent filters")
        elif strategy == "relax_quality_filters":
            reason_parts.append("returned after relaxing rating/count filters")
        elif strategy == "relax_multi_genre_to_primary_genre":
            reason_parts.append("returned after relaxing multi-genre requirement")
        elif strategy == "broad_semantic_fallback":
            reason_parts.append("returned by broad semantic fallback")

        exclude_genre = parsed_intent.get("exclude_genre")
        if exclude_genre:
            reason_parts.append(f"excluded genre: {exclude_genre}")

        if movie.get("ml_rating_mean") is not None:
            reason_parts.append(
                f"MovieLens mean rating {round(movie['ml_rating_mean'], 2)} "
                f"from {movie.get('ml_rating_count', 0)} ratings"
            )

        if movie.get("final_score") is not None:
            reason_parts.append(
                f"retrieval final score {movie['final_score']}"
            )

        return "; ".join(reason_parts)

    def build_agent_response(
        self,
        parsed_intent: dict,
        attempts: list[dict],
        fallback_used: bool,
        final_results: list[dict],
    ) -> str:
        if not final_results:
            return (
                "没有找到符合条件的电影。你可以尝试放宽年份、评分人数、评分要求，"
                "或者减少必须同时满足的类型条件。"
            )

        filter_parts = []

        if parsed_intent.get("genre"):
            filter_parts.append(f"类型为 {parsed_intent['genre']}")

        if parsed_intent.get("exclude_genre"):
            filter_parts.append(f"排除 {parsed_intent['exclude_genre']}")

        if parsed_intent.get("year_from"):
            filter_parts.append(f"{parsed_intent['year_from']} 年之后")

        if parsed_intent.get("min_vote_average") is not None:
            filter_parts.append(f"评分不低于 {parsed_intent['min_vote_average']}")

        if parsed_intent.get("min_vote_count") is not None:
            filter_parts.append(f"至少 {parsed_intent['min_vote_count']} 人评分")

        filter_text = "、".join(filter_parts)

        strict_result_count = 0
        if attempts:
            strict_result_count = attempts[0].get("result_count", 0)

        if fallback_used:
            fallback_text = (
                f"严格条件下找到 {strict_result_count} 部电影，所以系统自动放宽了部分条件，"
                "并使用多策略召回补充候选。"
            )
        else:
            fallback_text = "系统直接使用严格条件找到了足够的结果。"

        top_titles = [
            movie["title"]
            for movie in final_results
        ]

        return (
            f"我根据你的需求解析出筛选条件：{filter_text}。"
            f"{fallback_text}"
            f"最终推荐的是：{', '.join(top_titles)}。"
            "排序综合考虑了语义相关性、类型匹配、TMDB 质量分和 MovieLens 评分统计。"
        )

    def add_candidates(
        self,
        merged: dict,
        search_result: dict,
        strategy: str,
        parsed_intent: dict,
    ):
        required_genres = self.parse_genre_list(parsed_intent.get("genre"))

        for movie in search_result["results"]:
            movie_id = movie["id"]

            matched_required_genres = self.get_matched_required_genres(
                movie.get("genres"),
                required_genres,
            )

            required_match_count = len(matched_required_genres)
            required_total_count = len(required_genres)

            adjustment = self.get_strategy_adjustment(strategy)

            # 类型匹配加权：
            # - 完整匹配所有用户要求的 genre：明显加分
            # - 部分匹配：小幅加分
            # - 完全不匹配：明显扣分，避免 broad fallback 跑偏
            genre_match_bonus = 0.0

            if required_total_count > 0:
                if required_match_count == required_total_count:
                    genre_match_bonus = 0.15
                elif required_match_count > 0:
                    genre_match_bonus = 0.04
                else:
                    genre_match_bonus = -0.20

            agent_adjusted_score = movie["final_score"] + adjustment + genre_match_bonus

            movie["agent_strategy"] = strategy
            movie["matched_required_genres"] = matched_required_genres
            movie["required_genre_match_count"] = required_match_count
            movie["required_genre_total_count"] = required_total_count
            movie["genre_match_bonus"] = round(genre_match_bonus, 4)
            movie["agent_adjusted_score"] = round(agent_adjusted_score, 4)

            if movie_id not in merged:
                merged[movie_id] = movie
            else:
                if movie["agent_adjusted_score"] > merged[movie_id]["agent_adjusted_score"]:
                    merged[movie_id] = movie

    def normalize_parsed_intent(self, message: str, parsed_intent: dict) -> dict:
        """
        统一清洗规则抽参 / 本地 LLM 抽参结果。
        防止 local LLM 少字段、字段为字符串、top_k 为空等问题。
        """
        if parsed_intent is None:
            parsed_intent = {}

        normalized = {
            "query": parsed_intent.get("query") or message,
            "genre": parsed_intent.get("genre"),
            "exclude_genre": parsed_intent.get("exclude_genre"),
            "year_from": parsed_intent.get("year_from"),
            "year_to": parsed_intent.get("year_to"),
            "min_vote_average": parsed_intent.get("min_vote_average"),
            "min_vote_count": parsed_intent.get("min_vote_count"),
            "top_k": parsed_intent.get("top_k") or 5,
        }

        try:
            normalized["top_k"] = int(normalized["top_k"])
        except Exception:
            normalized["top_k"] = 5

        if normalized["top_k"] <= 0:
            normalized["top_k"] = 5

        if normalized["top_k"] > 20:
            normalized["top_k"] = 20

        for key in ["year_from", "year_to", "min_vote_count"]:
            if normalized[key] is not None:
                try:
                    normalized[key] = int(normalized[key])
                except Exception:
                    normalized[key] = None

        if normalized["min_vote_average"] is not None:
            try:
                normalized["min_vote_average"] = float(normalized["min_vote_average"])
            except Exception:
                normalized["min_vote_average"] = None

        return normalized

    def recommend(self, message: str) -> dict:
        parsed_intent = self.intent_parser_service.parse_recommend_intent(message)

        return self.recommend_with_parsed_intent(
            message=message,
            parsed_intent=parsed_intent,
            agent_type="rule_based_agent",
        )

    def recommend_with_parsed_intent(
            self,
            message: str,
            parsed_intent: dict,
            agent_type: str = "local_llm_agent",
    ) -> dict:
        parsed_intent = self.normalize_parsed_intent(
            message=message,
            parsed_intent=parsed_intent,
        )

        attempts = []
        merged = {}

        top_k = parsed_intent["top_k"]

        # 1. strict
        search_result = self.retrieval_service.search_movies(
            query=parsed_intent["query"],
            top_k=200,
            genre=parsed_intent["genre"],
            exclude_genre=parsed_intent["exclude_genre"],
            year_from=parsed_intent["year_from"],
            year_to=parsed_intent["year_to"],
            min_vote_average=parsed_intent["min_vote_average"],
            min_vote_count=parsed_intent["min_vote_count"],
        )

        attempts.append({
            "strategy": "strict",
            "genre": parsed_intent["genre"],
            "exclude_genre": parsed_intent["exclude_genre"],
            "year_from": parsed_intent["year_from"],
            "year_to": parsed_intent["year_to"],
            "min_vote_average": parsed_intent["min_vote_average"],
            "min_vote_count": parsed_intent["min_vote_count"],
            "result_count": len(search_result["results"]),
        })

        self.add_candidates(
            merged=merged,
            search_result=search_result,
            strategy="strict",
            parsed_intent=parsed_intent,
        )

        # 2. relax quality filters
        if len(merged) < top_k:
            relaxed_min_vote_average = parsed_intent["min_vote_average"]
            if relaxed_min_vote_average is None or relaxed_min_vote_average > 6:
                relaxed_min_vote_average = 6

            relaxed_min_vote_count = parsed_intent["min_vote_count"]
            if relaxed_min_vote_count is None or relaxed_min_vote_count > 150:
                relaxed_min_vote_count = 150

            search_result = self.retrieval_service.search_movies(
                query=parsed_intent["query"],
                top_k=200,
                genre=parsed_intent["genre"],
                exclude_genre=parsed_intent["exclude_genre"],
                year_from=parsed_intent["year_from"],
                year_to=parsed_intent["year_to"],
                min_vote_average=relaxed_min_vote_average,
                min_vote_count=relaxed_min_vote_count,
            )

            attempts.append({
                "strategy": "relax_quality_filters",
                "genre": parsed_intent["genre"],
                "exclude_genre": parsed_intent["exclude_genre"],
                "year_from": parsed_intent["year_from"],
                "year_to": parsed_intent["year_to"],
                "min_vote_average": relaxed_min_vote_average,
                "min_vote_count": relaxed_min_vote_count,
                "result_count": len(search_result["results"]),
            })

            self.add_candidates(
                merged=merged,
                search_result=search_result,
                strategy="relax_quality_filters",
                parsed_intent=parsed_intent,
            )

        # 3. relax multi-genre to primary genre
        required_genres = self.parse_genre_list(parsed_intent.get("genre"))

        if len(merged) < top_k and len(required_genres) >= 2:
            primary_genre = required_genres[0]

            search_result = self.retrieval_service.search_movies(
                query=parsed_intent["query"],
                top_k=200,
                genre=primary_genre,
                exclude_genre=parsed_intent["exclude_genre"],
                year_from=parsed_intent["year_from"],
                year_to=parsed_intent["year_to"],
                min_vote_average=6,
                min_vote_count=150,
            )

            attempts.append({
                "strategy": "relax_multi_genre_to_primary_genre",
                "genre": primary_genre,
                "exclude_genre": parsed_intent["exclude_genre"],
                "year_from": parsed_intent["year_from"],
                "year_to": parsed_intent["year_to"],
                "min_vote_average": 6,
                "min_vote_count": 150,
                "result_count": len(search_result["results"]),
            })

            self.add_candidates(
                merged=merged,
                search_result=search_result,
                strategy="relax_multi_genre_to_primary_genre",
                parsed_intent=parsed_intent,
            )

        # 4. broad semantic fallback
        if len(merged) < top_k:
            search_result = self.retrieval_service.search_movies(
                query=parsed_intent["query"],
                top_k=200,
                genre=None,
                exclude_genre=parsed_intent["exclude_genre"],
                year_from=parsed_intent["year_from"],
                year_to=parsed_intent["year_to"],
                min_vote_average=6,
                min_vote_count=150,
            )

            attempts.append({
                "strategy": "broad_semantic_fallback",
                "genre": None,
                "exclude_genre": parsed_intent["exclude_genre"],
                "year_from": parsed_intent["year_from"],
                "year_to": parsed_intent["year_to"],
                "min_vote_average": 6,
                "min_vote_count": 150,
                "result_count": len(search_result["results"]),
            })

            self.add_candidates(
                merged=merged,
                search_result=search_result,
                strategy="broad_semantic_fallback",
                parsed_intent=parsed_intent,
            )

        results = list(merged.values())

        required_genres = self.parse_genre_list(parsed_intent.get("genre"))
        required_genre_total = len(required_genres)

        strategy_priority = {
            "strict": 4,
            "relax_quality_filters": 3,
            "relax_multi_genre_to_primary_genre": 2,
            "broad_semantic_fallback": 1,
        }

        def ranking_key(movie: dict):
            required_match_count = movie.get("required_genre_match_count", 0)

            # 是否完整匹配用户要求的所有类型
            full_genre_match = 0
            if required_genre_total > 0 and required_match_count >= required_genre_total:
                full_genre_match = 1

            # fallback 策略优先级
            strategy_score = strategy_priority.get(
                movie.get("agent_strategy"),
                0
            )

            return (
                full_genre_match,  # 先保证完整类型匹配靠前
                required_match_count,  # 再看匹配了几个类型
                strategy_score,  # 再看来自哪个策略
                movie.get("agent_adjusted_score", 0) or 0,  # 再看 Agent 调整分
                movie.get("quality_score", 0) or 0,  # 再看质量分
                movie.get("ml_rating_count", 0) or 0,  # 最后看 MovieLens 评分人数
            )

        results.sort(key=ranking_key, reverse=True)

        final_results = results[:top_k]

        for movie in final_results:
            movie["agent_reason"] = self.build_agent_reason(
                movie=movie,
                parsed_intent=parsed_intent,
            )

        fallback_used = any(
            attempt["strategy"] != "strict"
            for attempt in attempts
        )

        agent_response = self.build_agent_response(
            parsed_intent=parsed_intent,
            attempts=attempts,
            fallback_used=fallback_used,
            final_results=final_results,
        )

        return {
            "message": message,
            "agent_type": agent_type,

            "agent_pipeline": [
                {
                    "agent": "IntentAgent",
                    "service": "LocalLLMService" if agent_type == "local_llm_agent" else "IntentParserService",
                    "role": "Parse natural language request into structured recommendation parameters",
                    "output": "extracted_params"
                },
                {
                    "agent": "RetrievalAgent",
                    "service": "RetrievalService",
                    "role": "Retrieve real movie candidates from Qdrant vector search and BM25 keyword search",
                    "output": "retrieved_candidates"
                },
                {
                    "agent": "FallbackAgent",
                    "service": "AgentRecommendationService",
                    "role": "Run strict search and fallback strategies when strict filters return insufficient results",
                    "output": "attempts"
                },
                {
                    "agent": "RankingAgent",
                    "service": "AgentRecommendationService",
                    "role": "Merge, deduplicate, and rerank candidates by semantic relevance, genre match, and quality scores",
                    "output": "ranked_results"
                },
                {
                    "agent": "ExplanationAgent",
                    "service": "AgentRecommendationService",
                    "role": "Generate recommendation reasons and final natural-language response",
                    "output": "agent_response"
                }
            ],

            "rag_pipeline": {
                "retrieval_sources": [
                    "Qdrant vector search",
                    "BM25 keyword search",
                    "TMDB movie metadata",
                    "MovieLens rating statistics"
                ],
                "retrieval_augmented_generation": True,
                "llm_does_not_generate_movie_titles": True,
                "description": (
                    "The LLM only extracts user intent. Movie candidates are retrieved from the local movie database, "
                    "then reranked and explained by the recommendation agent."
                )
            },

            "agent_response": agent_response,
            "parsed_intent": parsed_intent,
            "extracted_params": parsed_intent,
            "tool_called": "retrieval_service.search_movies",
            "attempts": attempts,
            "fallback_used": fallback_used,
            "candidate_count": len(results),
            "results": final_results,
        }

    def recommend_for_user(self, user_id: int, message: str) -> dict:
        """
        原来的规则解析版个性化 Agent 接口。
        """
        parsed_intent = self.intent_parser_service.parse_recommend_intent(message)

        return self.recommend_for_user_with_parsed_intent(
            user_id=user_id,
            message=message,
            parsed_intent=parsed_intent,
            agent_type="rule_based_personalized_agent",
        )

    def recommend_for_user_with_parsed_intent(
            self,
            user_id: int,
            message: str,
            parsed_intent: dict,
            agent_type: str = "local_llm_personalized_agent",
    ) -> dict:
        """
        接收已经解析好的结构化参数，执行：
        用户画像读取、已看过滤、fallback 检索、个性化重排和解释生成。
        """
        parsed_intent = self.normalize_parsed_intent( # 标准化，将所有的结果整合成一样的，调用前面的函数
            message=message,
            parsed_intent=parsed_intent,
        )

        if self.user_profile_service is None:
            return {
                "user_id": user_id,
                "message": message,
                "error": "user_profile_service is not available",
                "results": []
            }

        profile = self.user_profile_service.get_profile(user_id)

        if profile is None:
            return {
                "user_id": user_id,
                "message": message,
                "error": "User profile not found.",
                "results": []
            }

        if self.user_seen_movies_service is not None:
            seen_movie_ids = self.user_seen_movies_service.get_seen_movie_ids(user_id)
        else:
            seen_movie_ids = set()

        preferred_genres_text = profile.get("preferred_genres") or ""

        preferred_genres = [
            genre.strip() # 去空格
            for genre in preferred_genres_text.split(",")
            if genre.strip()
        ]

        # 如果用户自然语言里已经指定 genre，就优先尊重用户请求
        # 如果没指定，就用用户画像里的前 3 个偏好类型
        requested_genres = self.parse_genre_list(parsed_intent.get("genre"))

        if requested_genres:
            search_genres = [parsed_intent.get("genre")]
        else:
            search_genres = preferred_genres[:3] if preferred_genres else [None]

        enhanced_query = parsed_intent["query"]

        # 如果用户已经明确指定 genre，例如“科幻惊悚片”，
        # 不要把用户历史偏好拼进检索 query，避免语义召回跑偏到 Drama / Crime / Romance。
        # 用户画像只用于后面的弱排序加分。
        if preferred_genres and not requested_genres:
            enhanced_query = (
                    parsed_intent["query"]
                    + ". User historical preferred genres: "
                    + ", ".join(preferred_genres)
            )

        attempts = []
        merged = {}

        top_k = parsed_intent["top_k"]

        def add_user_candidates(search_result: dict, matched_genre: str | None, strategy: str):
            for movie in search_result["results"]:
                movie_id = movie["id"]

                if movie_id in seen_movie_ids:
                    continue

                # 如果用户明确指定了类型，最终结果必须至少匹配其中一个请求类型
                if requested_genres:
                    movie_genres = movie.get("genres") or ""
                    movie_genres_lower = movie_genres.lower() # 转小写

                    if not any(
                            requested_genre.lower() in movie_genres_lower
                            for requested_genre in requested_genres
                    ):
                        continue

                movie_genres = movie.get("genres") or ""
                movie_genres_lower = movie_genres.lower()

                matched_requested_genres = self.get_matched_required_genres(
                    movie_genres=movie_genres,
                    required_genres=requested_genres,
                )

                # 用户历史偏好加分
                user_genre_boost = 0.0

                for rank, preferred_genre in enumerate(preferred_genres[:5]):
                    if preferred_genre.lower() in movie_genres_lower:
                        user_genre_boost = max(
                            user_genre_boost,
                            max(0.12 - rank * 0.02, 0.02)
                        )

                # 如果用户显式指定了请求类型，例如 Science Fiction,Thriller，
                # 用户历史偏好只能作为弱信号，不能盖过用户当前需求。
                if requested_genres:
                    user_genre_boost = user_genre_boost * 0.3

                # 用户当前明确请求的类型优先级要高于历史画像
                request_genre_bonus = 0.0

                if requested_genres:
                    if len(matched_requested_genres) == len(requested_genres):
                        request_genre_bonus = 0.18
                    elif len(matched_requested_genres) > 0:
                        request_genre_bonus = 0.05

                # fallback 策略惩罚，避免宽松结果超过严格匹配结果
                strategy_bonus = {
                    "personalized_strict": 0.10,
                    "personalized_relax_quality": 0.06,
                    "personalized_relax_multi_genre": -0.05,
                    "personalized_broad_fallback": -0.15,
                }.get(strategy, 0.0)

                # 当前请求优先级 > 用户历史偏好
                # 如果用户明确要求多个 genre：
                # - 全部匹配：强加分
                # - 部分匹配：小加分
                # - broad fallback：明显降权
                if requested_genres:
                    if len(matched_requested_genres) == len(requested_genres):
                        request_genre_bonus = 0.30
                    elif len(matched_requested_genres) > 0:
                        request_genre_bonus = 0.03
                    else:
                        request_genre_bonus = -0.30

                # 用户画像只作为辅助信号，不能压过当前自然语言请求
                user_genre_boost = min(user_genre_boost, 0.06)

                agent_adjusted_score = (
                        movie["final_score"]
                        + request_genre_bonus # 类型
                        + user_genre_boost # 偏好
                        + strategy_bonus # 策略
                )

                movie["agent_strategy"] = strategy
                movie["matched_user_genre"] = matched_genre
                movie["matched_requested_genres"] = matched_requested_genres
                movie["requested_genre_match_count"] = len(matched_requested_genres)
                movie["requested_genre_total_count"] = len(requested_genres)
                movie["user_genre_boost"] = round(user_genre_boost, 4)
                movie["request_genre_bonus"] = round(request_genre_bonus, 4)
                movie["strategy_bonus"] = round(strategy_bonus, 4)
                movie["agent_adjusted_score"] = round(agent_adjusted_score, 4)

                reason_parts = []

                if matched_requested_genres:
                    if len(matched_requested_genres) == len(requested_genres):
                        reason_parts.append(
                            "matched all requested genres: "
                            + ", ".join(matched_requested_genres)
                        )
                    else:
                        reason_parts.append(
                            "partially matched requested genres: "
                            + ", ".join(matched_requested_genres)
                        )

                if matched_genre:
                    reason_parts.append(
                        f"retrieved using genre condition: {matched_genre}"
                    )

                if user_genre_boost > 0:
                    reason_parts.append(
                        "boosted by user's historical genre preference"
                    )

                if strategy == "personalized_strict":
                    reason_parts.append("returned by personalized strict retrieval")
                elif strategy == "personalized_relax_quality":
                    reason_parts.append("returned after relaxing quality filters")
                elif strategy == "personalized_relax_multi_genre":
                    reason_parts.append("returned after relaxing multi-genre requirement")
                elif strategy == "personalized_broad_fallback":
                    reason_parts.append("returned by personalized broad fallback")

                if movie.get("ml_rating_mean") is not None:
                    reason_parts.append(
                        f"MovieLens mean rating {round(movie['ml_rating_mean'], 2)} "
                        f"from {movie.get('ml_rating_count', 0)} ratings"
                    )

                reason_parts.append("excluded movies already rated by the user")

                movie["agent_reason"] = "; ".join(reason_parts)

                if movie_id not in merged:
                    merged[movie_id] = movie
                else:
                    if movie["agent_adjusted_score"] > merged[movie_id]["agent_adjusted_score"]:
                        merged[movie_id] = movie

        # 第一轮：按请求 genre 或用户偏好 genre 召回
        for genre in search_genres:
            search_result = self.retrieval_service.search_movies(
                query=enhanced_query,
                top_k=200,
                genre=genre,
                exclude_genre=parsed_intent["exclude_genre"],
                year_from=parsed_intent["year_from"],
                year_to=parsed_intent["year_to"],
                min_vote_average=parsed_intent["min_vote_average"],
                min_vote_count=parsed_intent["min_vote_count"],
            )

            attempts.append({
                "strategy": "personalized_strict",
                "genre": genre,
                "exclude_genre": parsed_intent["exclude_genre"],
                "min_vote_average": parsed_intent["min_vote_average"],
                "min_vote_count": parsed_intent["min_vote_count"],
                "result_count": len(search_result["results"]),
            })

            add_user_candidates(
                search_result=search_result,
                matched_genre=genre,
                strategy="personalized_strict",
            )

        # 第二轮：如果不够，放宽质量门槛
        if len(merged) < top_k:
            for genre in search_genres:
                search_result = self.retrieval_service.search_movies(
                    query=enhanced_query,
                    top_k=200,
                    genre=genre,
                    exclude_genre=parsed_intent["exclude_genre"],
                    year_from=parsed_intent["year_from"],
                    year_to=parsed_intent["year_to"],
                    min_vote_average=None,
                    min_vote_count=None,
                )

                attempts.append({
                    "strategy": "personalized_relax_quality",
                    "genre": genre,
                    "exclude_genre": parsed_intent["exclude_genre"],
                    "min_vote_average": 6.0,
                    "min_vote_count": 100,
                    "result_count": len(search_result["results"]),
                })

                add_user_candidates(
                    search_result=search_result,
                    matched_genre=genre,
                    strategy="personalized_relax_quality",
                )

        # 第三轮：如果还不够，兜底召回
        # 注意：如果用户明确指定了 genre，不要完全取消 genre，避免推荐跑偏
        # 第三轮：如果用户要求多个类型，而严格 AND 结果不够，
        # 则放宽为单类型分别召回。
        if len(merged) < top_k and len(requested_genres) >= 2:
            for genre in requested_genres:
                search_result = self.retrieval_service.search_movies(
                    query=enhanced_query,
                    top_k=200,
                    genre=genre,
                    exclude_genre=parsed_intent["exclude_genre"],
                    year_from=parsed_intent["year_from"],
                    year_to=parsed_intent["year_to"],
                    min_vote_average=6.0,
                    min_vote_count=100,
                )

                attempts.append({
                    "strategy": "personalized_relax_multi_genre",
                    "genre": genre,
                    "exclude_genre": parsed_intent["exclude_genre"],
                    "year_from": parsed_intent["year_from"],
                    "year_to": parsed_intent["year_to"],
                    "min_vote_average": 6.0,
                    "min_vote_count": 100,
                    "result_count": len(search_result["results"]),
                })

                add_user_candidates(
                    search_result=search_result,
                    matched_genre=genre,
                    strategy="personalized_relax_multi_genre",
                )

        # 第四轮：如果仍然不足，进行宽松语义兜底召回。
        # add_user_candidates() 会保证：
        # - 已看电影不会返回
        # - 如果用户明确指定了 genre，最终结果至少匹配其中一个类型
        if len(merged) < top_k:
            search_result = self.retrieval_service.search_movies(
                query=enhanced_query,
                top_k=200,
                genre=None,
                exclude_genre=parsed_intent["exclude_genre"],
                year_from=parsed_intent["year_from"],
                year_to=parsed_intent["year_to"],
                min_vote_average=None,
                min_vote_count=30,
            )

            attempts.append({
                "strategy": "personalized_broad_fallback",
                "genre": None,
                "exclude_genre": parsed_intent["exclude_genre"],
                "year_from": parsed_intent["year_from"],
                "year_to": parsed_intent["year_to"],
                "min_vote_average": None,
                "min_vote_count": 30,
                "result_count": len(search_result["results"]),
            })

            add_user_candidates(
                search_result=search_result,
                matched_genre=None,
                strategy="personalized_broad_fallback",
            )

        results = list(merged.values())

        requested_genres = self.parse_genre_list(
            parsed_intent.get("genre")
        )

        def get_rank_score(movie: dict) -> float:
            return float(
                movie.get("personalized_score")
                or movie.get("agent_adjusted_score")
                or movie.get("final_score")
                or 0
            )

        def get_quality_bucket(movie: dict) -> int:
            """
            质量分层：
            3 = 比较可靠的高质量候选
            2 = 可接受候选
            1 = 勉强可作为补位
            0 = 低质量补位
            """
            vote_average = float(movie.get("vote_average") or 0)
            vote_count = int(movie.get("vote_count") or 0)
            ml_rating_mean = float(movie.get("ml_rating_mean") or 0)
            ml_rating_count = int(movie.get("ml_rating_count") or 0)

            if (vote_average >= 6.8 or ml_rating_mean >= 4.0 ) and (vote_count >= 500 or ml_rating_count >= 400 ):
                return 3

            if (vote_average >= 6.0 or ml_rating_mean >= 3.5 ) and (
                    vote_count >= 100 or ml_rating_count >= 80
            ):
                return 2

            if (vote_average >= 4.0 or ml_rating_mean >= 2.0 ) and (
                    vote_count >= 50 or ml_rating_count >= 30
            ):
                return 1

            return 0

        def get_strategy_priority(movie: dict) -> int:
            strategy = movie.get("agent_strategy")

            if strategy == "personalized_strict":
                return 5

            if strategy == "personalized_relax_quality":
                return 4

            if strategy == "personalized_relax_multi_genre":
                return 3

            if strategy == "personalized_genre_preserving_fallback":
                return 2

            if strategy == "personalized_broad_fallback":
                return 1

            return 0

        def is_full_requested_genre_match(movie: dict) -> bool:
            if not requested_genres:
                return True

            match_count = movie.get("requested_genre_match_count", 0)
            total_count = movie.get(
                "requested_genre_total_count",
                len(requested_genres)
            )

            return match_count >= total_count

        results.sort(
            key=lambda movie: (
                1 if is_full_requested_genre_match(movie) else 0,
                get_quality_bucket(movie),
                get_strategy_priority(movie),
                get_rank_score(movie),
                movie.get("quality_score") or 0,
                movie.get("ml_rating_mean") or 0,
                movie.get("ml_rating_count") or 0,
            ),
            reverse=True
        )

        acceptable_results = [
            movie
            for movie in results
            if get_quality_bucket(movie) >= 1
        ]

        low_quality_results = [
            movie
            for movie in results
            if get_quality_bucket(movie) == 0
        ]

        if len(acceptable_results) >= top_k:
            final_results = acceptable_results[:top_k]
        else:
            final_results = (
                                    acceptable_results
                                    + low_quality_results
                            )[:top_k]

        for movie in final_results:
            quality_bucket = get_quality_bucket(movie)

            movie["quality_bucket"] = quality_bucket
            movie["is_low_quality_fallback"] = quality_bucket == 0
            movie["is_weak_fallback"] = quality_bucket <= 1

            if quality_bucket >= 3:
                movie["recommendation_confidence"] = "high"
                movie["recommendation_level"] = "strong_match"
            elif quality_bucket == 2:
                movie["recommendation_confidence"] = "medium"
                movie["recommendation_level"] = "acceptable_match"
            elif quality_bucket == 1:
                movie["recommendation_confidence"] = "low"
                movie["recommendation_level"] = "weak_fallback"
            else:
                movie["recommendation_confidence"] = "very_low"
                movie["recommendation_level"] = "low_quality_fallback"

        if final_results:
            titles = [movie["title"] for movie in final_results]

            requested_count = top_k
            returned_count = len(final_results)

            strict_count = sum(
                attempt.get("result_count", 0)
                for attempt in attempts
                if attempt.get("strategy") == "personalized_strict"
            )

            fallback_attempts = [
                attempt
                for attempt in attempts
                if attempt.get("strategy") != "personalized_strict"
            ]

            fallback_used = len(fallback_attempts) > 0

            low_quality_count = sum(
                1
                for movie in final_results
                if movie.get("is_low_quality_fallback")
            )

            weak_fallback_count = sum(
                1
                for movie in final_results
                if movie.get("is_weak_fallback")
            )

            request_parts = []

            if parsed_intent.get("genre"):
                request_parts.append(f"类型包含 {parsed_intent['genre']}")

            if parsed_intent.get("exclude_genre"):
                request_parts.append(f"排除 {parsed_intent['exclude_genre']}")

            if parsed_intent.get("year_from"):
                request_parts.append(f"{parsed_intent['year_from']} 年之后")

            if parsed_intent.get("year_to"):
                request_parts.append(f"{parsed_intent['year_to']} 年之前")

            if parsed_intent.get("min_vote_average") is not None:
                request_parts.append(f"评分不低于 {parsed_intent['min_vote_average']}")

            if parsed_intent.get("min_vote_count") is not None:
                request_parts.append(f"至少 {parsed_intent['min_vote_count']} 人评分")

            request_text = "、".join(request_parts) if request_parts else "没有明确筛选条件"

            if returned_count < requested_count:
                count_text = (
                    f"你请求推荐 {requested_count} 部电影，但在排除该用户已评分电影并应用筛选条件后，"
                    f"系统只找到 {returned_count} 部可推荐的新候选。"
                )
            else:
                count_text = f"系统返回了 {returned_count} 部推荐结果。"

            if strict_count == 0 and fallback_used:
                fallback_text = (
                    "严格条件下没有找到足够的新电影，因此系统自动放宽了评分、评分人数或多类型匹配要求，"
                    "并使用 fallback 策略补充候选。"
                )
            elif strict_count < requested_count and fallback_used:
                fallback_text = (
                    f"严格条件下只找到 {strict_count} 部候选，数量不足，"
                    "因此系统使用 fallback 策略补充了部分相关电影。"
                )
            else:
                fallback_text = "这些结果主要来自严格筛选条件。"

            if low_quality_count > 0:
                quality_text = (
                    f"其中有 {low_quality_count} 部属于低质量补位结果，"
                    "主要用于在候选严重不足时补足推荐数量，建议谨慎参考。"
                )
            elif weak_fallback_count > 0:
                quality_text = (
                    f"其中有 {weak_fallback_count} 部属于弱补位结果，"
                    "它们满足部分类型或语义相关性，但评分热度或质量稳定性不如排序靠前的候选。"
                )
            else:
                quality_text = "最终结果已尽量优先保留类型匹配和质量较稳定的电影。"

            agent_response = (
                f"我先将你的自然语言需求解析为：{request_text}。"
                f"同时结合用户 {user_id} 的历史偏好进行个性化推荐，"
                f"该用户偏好的类型包括：{preferred_genres_text or '暂无'}。"
                f"系统已排除该用户评分过的 {len(seen_movie_ids)} 部电影。"
                f"{count_text}"
                f"{fallback_text}"
                f"{quality_text}"
                f"最终推荐的是：{', '.join(titles)}。"
                "排序综合考虑了语义相关性、请求类型匹配、用户偏好类型、TMDB 质量分和 MovieLens 评分统计。"
            )
        else:
            agent_response = (
                f"没有为用户 {user_id} 找到合适的新电影。"
                "当前条件可能过于严格，建议放宽评分、评分人数、年份范围，"
                "或者减少必须同时满足的电影类型。"
            )

        fallback_used = any(
            attempt["strategy"] != "personalized_strict"
            for attempt in attempts
        )

        return {
            "user_id": user_id,
            "message": message,
            "agent_type": agent_type,

            "agent_pipeline": [
                {
                    "agent": "IntentAgent",
                    "service": (
                        "LocalLLMService"
                        if agent_type == "local_llm_personalized_agent"
                        else "IntentParserService"
                    ),
                    "role": "Parse natural language into structured recommendation parameters"
                },
                {
                    "agent": "PersonalizationAgent",
                    "service": "UserProfileService",
                    "role": "Load user genre preferences from historical ratings"
                },
                {
                    "agent": "SeenFilterAgent",
                    "service": "UserSeenMoviesService",
                    "role": "Exclude movies already rated by the user"
                },
                {
                    "agent": "RetrievalAgent",
                    "service": "RetrievalService",
                    "role": "Retrieve real movie candidates using vector search and BM25"
                },
                {
                    "agent": "FallbackAgent",
                    "service": "AgentRecommendationService",
                    "role": "Run personalized fallback retrieval strategies"
                },
                {
                    "agent": "RankingAgent",
                    "service": "AgentRecommendationService",
                    "role": "Rerank by request match, user preference and quality scores"
                }
            ],

            "agent_response": agent_response,
            "parsed_intent": parsed_intent,
            "extracted_params": parsed_intent,
            "profile": profile,
            "used_preferred_genres": preferred_genres[:3],
            "tool_called": "retrieval_service.search_movies",
            "attempts": attempts,
            "fallback_used": fallback_used,
            "exclude_seen": True,
            "seen_movie_count": len(seen_movie_ids),
            "candidate_count": len(results),
            "requested_count": top_k,
            "returned_count": len(final_results),
            "results": final_results,
        }
