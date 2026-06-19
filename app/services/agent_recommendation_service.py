import logging

logger = logging.getLogger(__name__)


class AgentRecommendationService:
    def __init__(
        self,
        llm_service,           # LLMService 实例（cloud 或 local）
        retrieval_service,     # RetrievalService 实例
        intent_parser_service=None,  # 规则意图解析器（fallback）
        user_profile_service=None,
        user_seen_movies_service=None,
    ):
        self.llm_service = llm_service
        self.retrieval_service = retrieval_service
        self.intent_parser_service = intent_parser_service
        self.user_profile_service = user_profile_service
        self.user_seen_movies_service = user_seen_movies_service

    # ──────────────────── 工具函数（保留） ────────────────────

    def parse_genre_list(self, value: str | None) -> list[str]:
        if not value:
            return []
        return [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

    def get_matched_required_genres(
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
        mapping = {
            "strict": 0.08,
            "relax_quality_filters": 0.06,
            "relax_multi_genre_to_primary_genre": -0.07,
            "broad_semantic_fallback": -0.10,
        }
        return mapping.get(strategy, 0.0)

    def normalize_parsed_intent(self, message: str, parsed_intent: dict) -> dict:
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
                movie.get("genres"), required_genres,
            )
            required_match_count = len(matched_required_genres)
            required_total_count = len(required_genres)
            adjustment = self.get_strategy_adjustment(strategy)

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

    # ──────────────────── Agent 决策循环（NEW） ────────────────────

    def _build_decision_context(
        self,
        user_message: str,
        current_params: dict,
        merged: dict,
        attempts: list[dict],
        turn: int,
        max_turns: int,
    ) -> str:
        """构建 Agent 决策的上下文文本。"""
        top_k = current_params.get("top_k", 5)
        candidate_count = len(merged)

        lines = [
            f"用户需求: {user_message}",
            f"当前轮次: 第 {turn + 1} 轮 / 共 {max_turns} 轮",
            f"目标推荐数量: {top_k}",
            f"当前已收集候选数: {candidate_count}",
            "",
            "当前搜索参数:",
            f"  query={current_params.get('query')}",
            f"  genre={current_params.get('genre')}",
            f"  exclude_genre={current_params.get('exclude_genre')}",
            f"  year_from={current_params.get('year_from')}",
            f"  year_to={current_params.get('year_to')}",
            f"  min_vote_average={current_params.get('min_vote_average')}",
            f"  min_vote_count={current_params.get('min_vote_count')}",
            "",
            "搜索历史:",
        ]

        for i, attempt in enumerate(attempts):
            lines.append(
                f"  尝试 {i + 1}: strategy={attempt.get('strategy')}, "
                f"genre={attempt.get('genre')}, "
                f"min_vote_average={attempt.get('min_vote_average')}, "
                f"min_vote_count={attempt.get('min_vote_count')}, "
                f"返回 {attempt.get('result_count', 0)} 条结果"
            )

        return "\n".join(lines)

    def _agent_loop(
        self,
        user_message: str,
        initial_intent: dict,
        user_id: int | None = None,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        LLM 驱动的 Agent 搜索循环。

        每轮让 LLM 根据当前候选情况决定：继续搜索（可能修改参数）还是结束。

        Returns:
            (merged_list, attempts, agent_trace)
        """
        max_turns = 5
        merged = {}
        attempts = []
        agent_trace = []
        current_params = initial_intent.copy()
        top_k = current_params.get("top_k", 5)

        # 第一轮：用初始参数搜索
        first_result = self.retrieval_service.search_movies(
            query=current_params["query"],
            top_k=200,
            genre=current_params.get("genre"),
            exclude_genre=current_params.get("exclude_genre"),
            year_from=current_params.get("year_from"),
            year_to=current_params.get("year_to"),
            min_vote_average=current_params.get("min_vote_average"),
            min_vote_count=current_params.get("min_vote_count"),
        )

        attempts.append({
            "strategy": "strict",
            "genre": current_params.get("genre"),
            "exclude_genre": current_params.get("exclude_genre"),
            "year_from": current_params.get("year_from"),
            "year_to": current_params.get("year_to"),
            "min_vote_average": current_params.get("min_vote_average"),
            "min_vote_count": current_params.get("min_vote_count"),
            "result_count": len(first_result["results"]),
        })

        self.add_candidates(merged, first_result, "strict", current_params)

        agent_trace.append({
            "turn": 1,
            "decision": "search_movies",
            "reasoning": "使用初始参数进行首轮严格搜索",
            "candidates_found": len(first_result["results"]),
            "total_candidates": len(merged),
        })

        # 如果第一轮已经够了，直接返回
        if len(merged) >= top_k:
            agent_trace.append({
                "turn": 1,
                "decision": "finalize",
                "reasoning": f"严格搜索已获得 {len(merged)} 部候选，满足需求",
                "candidates_found": len(first_result["results"]),
                "total_candidates": len(merged),
            })
            return list(merged.values()), attempts, agent_trace

        # 后续轮次：LLM 决策
        for turn in range(1, max_turns):
            decision_context = self._build_decision_context(
                user_message, current_params, merged, attempts, turn, max_turns,
            )

            try:
                decision = self.llm_service.decide_next_action(decision_context)
            except Exception as e:
                logger.warning(f"LLM decision failed at turn {turn + 1}: {e}, using rule fallback")
                break

            agent_trace.append({
                "turn": turn + 1,
                "decision": decision.get("action", "unknown"),
                "reasoning": decision.get("reasoning", ""),
                "llm_raw_decision": decision,
            })

            if decision.get("action") == "finalize":
                break

            if decision.get("action") == "search_movies":
                params = decision.get("params", {})
                search_params = {
                    "query": params.get("query", current_params["query"]),
                    "top_k": 200,
                    "genre": params.get("genre", current_params.get("genre")),
                    "exclude_genre": params.get("exclude_genre", current_params.get("exclude_genre")),
                    "year_from": params.get("year_from", current_params.get("year_from")),
                    "year_to": params.get("year_to", current_params.get("year_to")),
                    "min_vote_average": params.get("min_vote_average", current_params.get("min_vote_average")),
                    "min_vote_count": params.get("min_vote_count", current_params.get("min_vote_count")),
                }

                strategy = params.get("strategy", "llm_modified")
                search_result = self.retrieval_service.search_movies(**search_params)
                attempts.append({
                    "strategy": strategy,
                    **search_params,
                    "result_count": len(search_result["results"]),
                })

                self.add_candidates(merged, search_result, strategy, current_params)

                # 更新 agent_trace
                agent_trace[-1]["candidates_found"] = len(search_result["results"])
                agent_trace[-1]["total_candidates"] = len(merged)

                if len(merged) >= top_k:
                    agent_trace.append({
                        "turn": turn + 1,
                        "decision": "finalize",
                        "reasoning": f"候选已累积至 {len(merged)} 部，停止搜索",
                        "total_candidates": len(merged),
                    })
                    break
            else:
                # 未知 action，跳出
                logger.warning(f"Unknown agent action: {decision.get('action')}")
                break

        return list(merged.values()), attempts, agent_trace

    def _rule_based_fallback_loop(
        self,
        parsed_intent: dict,
        merged: dict,
        attempts: list[dict],
    ) -> list[dict]:
        """
        规则回退循环 —— 当 LLM 决策失败时使用。

        保留原有的硬编码四阶段逻辑作为安全网。
        """
        top_k = parsed_intent["top_k"]

        # 1. strict（已在 _agent_loop 第一轮执行）
        # 2. relax quality filters
        if len(merged) < top_k:
            relaxed_min_vote_average = parsed_intent["min_vote_average"]
            if relaxed_min_vote_average is None or relaxed_min_vote_average > 6:
                relaxed_min_vote_average = 6
            relaxed_min_vote_count = parsed_intent["min_vote_count"]
            if relaxed_min_vote_count is None or relaxed_min_vote_count > 150:
                relaxed_min_vote_count = 150

            search_result = self.retrieval_service.search_movies(
                query=parsed_intent["query"], top_k=200,
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
                "min_vote_average": relaxed_min_vote_average,
                "min_vote_count": relaxed_min_vote_count,
                "result_count": len(search_result["results"]),
            })
            self.add_candidates(merged, search_result, "relax_quality_filters", parsed_intent)

        # 3. relax multi-genre
        required_genres = self.parse_genre_list(parsed_intent.get("genre"))
        if len(merged) < top_k and len(required_genres) >= 2:
            primary_genre = required_genres[0]
            search_result = self.retrieval_service.search_movies(
                query=parsed_intent["query"], top_k=200,
                genre=primary_genre,
                exclude_genre=parsed_intent["exclude_genre"],
                year_from=parsed_intent["year_from"],
                year_to=parsed_intent["year_to"],
                min_vote_average=6, min_vote_count=150,
            )
            attempts.append({
                "strategy": "relax_multi_genre_to_primary_genre",
                "genre": primary_genre,
                "min_vote_average": 6, "min_vote_count": 150,
                "result_count": len(search_result["results"]),
            })
            self.add_candidates(merged, search_result, "relax_multi_genre_to_primary_genre", parsed_intent)

        # 4. broad semantic fallback
        if len(merged) < top_k:
            search_result = self.retrieval_service.search_movies(
                query=parsed_intent["query"], top_k=200,
                genre=None,
                exclude_genre=parsed_intent["exclude_genre"],
                year_from=parsed_intent["year_from"],
                year_to=parsed_intent["year_to"],
                min_vote_average=6, min_vote_count=150,
            )
            attempts.append({
                "strategy": "broad_semantic_fallback",
                "genre": None,
                "min_vote_average": 6, "min_vote_count": 150,
                "result_count": len(search_result["results"]),
            })
            self.add_candidates(merged, search_result, "broad_semantic_fallback", parsed_intent)

        return list(merged.values())

    # ──────────────────── 排序 ────────────────────

    def _rank_candidates(
        self,
        results: list[dict],
        parsed_intent: dict,
    ) -> list[dict]:
        """对候选集进行排序（确定性代码，非 LLM）。"""
        required_genres = self.parse_genre_list(parsed_intent.get("genre"))
        required_genre_total = len(required_genres)

        strategy_priority = {
            "strict": 4,
            "relax_quality_filters": 3,
            "relax_multi_genre_to_primary_genre": 2,
            "broad_semantic_fallback": 1,
            "llm_modified": 3,
        }

        def ranking_key(movie: dict):
            required_match_count = movie.get("required_genre_match_count", 0)
            full_genre_match = (
                1 if required_genre_total > 0 and required_match_count >= required_genre_total
                else 0
            )
            strategy_score = strategy_priority.get(movie.get("agent_strategy"), 0)

            return (
                full_genre_match,
                required_match_count,
                strategy_score,
                movie.get("agent_adjusted_score", 0) or 0,
                movie.get("quality_score", 0) or 0,
                movie.get("ml_rating_count", 0) or 0,
            )

        results.sort(key=ranking_key, reverse=True)
        return results

    # ──────────────────── 动态 Pipeline 生成 ────────────────────

    def _build_agent_pipeline(
        self,
        agent_type: str,
        used_rule_fallback: bool,
        attempts: list[dict],
    ) -> list[dict]:
        """根据实际执行情况动态生成 agent_pipeline。"""
        pipeline = [
            {
                "agent": "IntentAgent",
                "service": "LLMService" if "llm" in agent_type else "IntentParserService",
                "role": "Parse natural language request into structured recommendation parameters",
                "output": "extracted_params",
            },
        ]

        # 检索阶段
        strategies_used = set(a.get("strategy", "") for a in attempts)
        pipeline.append({
            "agent": "RetrievalAgent",
            "service": "RetrievalService",
            "role": (
                f"Retrieve movie candidates via Qdrant vector + BM25 keyword search. "
                f"Strategies used: {', '.join(strategies_used) if strategies_used else 'strict'}"
            ),
            "output": "retrieved_candidates",
            "attempts": len(attempts),
        })

        # 回退（如果有）
        if len(attempts) > 1:
            pipeline.append({
                "agent": "FallbackAgent",
                "service": "AgentRecommendationService" if used_rule_fallback else "LLMService",
                "role": (
                    "LLM-driven search loop with automatic fallback strategies"
                    if not used_rule_fallback
                    else "Rule-based fallback strategies (LLM decision failed)"
                ),
                "output": "attempts",
                "fallback_used": True,
                "llm_driven": not used_rule_fallback,
            })

        pipeline.extend([
            {
                "agent": "RankingAgent",
                "service": "AgentRecommendationService",
                "role": "Merge, deduplicate, and rerank candidates by genre match, strategy priority, and quality scores",
                "output": "ranked_results",
            },
            {
                "agent": "ExplanationAgent",
                "service": "LLMService",
                "role": "Generate natural-language recommendation explanations via RAG: LLM receives retrieved movie metadata as context and produces personalized recommendation reasons",
                "output": "agent_response",
                "rag_enabled": True,
            },
        ])

        return pipeline

    # ──────────────────── 公开接口 ────────────────────

    def recommend(self, message: str) -> dict:
        """规则版 Agent 接口（保持向后兼容）。"""
        if self.intent_parser_service is not None:
            parsed_intent = self.intent_parser_service.parse_recommend_intent(message)
        else:
            parsed_intent = self.llm_service.extract_search_params(message)

        return self.recommend_with_parsed_intent(
            message=message,
            parsed_intent=parsed_intent,
            agent_type="rule_based_agent",
        )

    def recommend_llm(self, message: str) -> dict:
        """LLM 版 Agent 接口 —— 使用 Agent 循环 + RAG 生成。"""
        # 意图解析（LLM 或规则回退）
        try:
            parsed_intent = self.llm_service.extract_search_params(message)
        except Exception:
            if self.intent_parser_service is not None:
                parsed_intent = self.intent_parser_service.parse_recommend_intent(message)
            else:
                raise

        return self.recommend_with_parsed_intent(
            message=message,
            parsed_intent=parsed_intent,
            agent_type="llm_agent",
            use_agent_loop=True,
        )

    def recommend_with_parsed_intent(
        self,
        message: str,
        parsed_intent: dict,
        agent_type: str = "llm_agent",
        use_agent_loop: bool = True,
    ) -> dict:
        """使用已解析的意图执行推荐（核心方法）。"""
        parsed_intent = self.normalize_parsed_intent(message, parsed_intent)
        top_k = parsed_intent["top_k"]

        merged = {}
        attempts = []
        agent_trace = []
        used_rule_fallback = False

        if use_agent_loop and agent_type == "llm_agent":
            # ─── LLM 驱动的 Agent 循环 ───
            try:
                results, attempts, agent_trace = self._agent_loop(message, parsed_intent)
            except Exception as e:
                logger.warning(f"LLM agent loop failed: {e}, falling back to rule-based")
                used_rule_fallback = True
                # 第一轮严格搜索 + 规则回退
                first_result = self.retrieval_service.search_movies(
                    query=parsed_intent["query"], top_k=200,
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
                    "min_vote_average": parsed_intent["min_vote_average"],
                    "min_vote_count": parsed_intent["min_vote_count"],
                    "result_count": len(first_result["results"]),
                })
                self.add_candidates(merged, first_result, "strict", parsed_intent)
                results = self._rule_based_fallback_loop(parsed_intent, merged, attempts)
                agent_trace = [{
                    "turn": 1,
                    "decision": "rule_fallback",
                    "reasoning": "LLM agent loop failed, using rule-based fallback",
                }]
        else:
            # ─── 规则回退（原有逻辑） ───
            used_rule_fallback = True
            first_result = self.retrieval_service.search_movies(
                query=parsed_intent["query"], top_k=200,
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
                "min_vote_average": parsed_intent["min_vote_average"],
                "min_vote_count": parsed_intent["min_vote_count"],
                "result_count": len(first_result["results"]),
            })
            self.add_candidates(merged, first_result, "strict", parsed_intent)
            results = self._rule_based_fallback_loop(parsed_intent, merged, attempts)

        # ─── 排序 ───
        results = self._rank_candidates(results, parsed_intent)
        final_results = results[:top_k]

        # ─── RAG 生成推荐回复 ───
        try:
            agent_response, individual_reasons = self.llm_service.generate_recommendation_response(
                user_message=message,
                parsed_intent=parsed_intent,
                candidates=final_results,
                user_profile=None,
                attempts=attempts,
                individual_reasons=True,
            )
            rag_generated = True
            # 逐条填入 LLM 生成的推荐理由
            for movie, reason in zip(final_results, individual_reasons):
                movie["agent_reason"] = reason
        except Exception as e:
            logger.warning(f"RAG generation failed: {e}, using template fallback")
            rag_generated = False
            agent_response = self._template_response(parsed_intent, attempts, final_results)
            for movie in final_results:
                movie["agent_reason"] = self._template_reason(movie, parsed_intent)

        # ─── 构建响应 ───
        fallback_used = any(a["strategy"] != "strict" for a in attempts)

        agent_pipeline = self._build_agent_pipeline(agent_type, used_rule_fallback, attempts)

        return {
            "message": message,
            "agent_type": agent_type,
            "agent_pipeline": agent_pipeline,
            "agent_trace": agent_trace,
            "rag_generated": rag_generated,
            "llm_driven_loop": use_agent_loop and not used_rule_fallback,

            "agent_response": agent_response,
            "parsed_intent": parsed_intent,
            "extracted_params": parsed_intent,
            "tool_called": "retrieval_service.search_movies",
            "attempts": attempts,
            "fallback_used": fallback_used,
            "rule_fallback_used": used_rule_fallback,
            "candidate_count": len(results),
            "results": final_results,
        }

    # ──────────────────── 个性化推荐（用户画像） ────────────────────

    def recommend_for_user(self, user_id: int, message: str) -> dict:
        """规则版个性化 Agent 接口。"""
        if self.intent_parser_service is not None:
            parsed_intent = self.intent_parser_service.parse_recommend_intent(message)
        else:
            parsed_intent = self.llm_service.extract_search_params(message)

        return self.recommend_for_user_with_parsed_intent(
            user_id=user_id, message=message,
            parsed_intent=parsed_intent,
            agent_type="rule_based_personalized_agent",
        )

    def recommend_for_user_llm(self, user_id: int, message: str) -> dict:
        """LLM 版个性化 Agent 接口。"""
        try:
            parsed_intent = self.llm_service.extract_search_params(message)
        except Exception:
            if self.intent_parser_service is not None:
                parsed_intent = self.intent_parser_service.parse_recommend_intent(message)
            else:
                raise

        return self.recommend_for_user_with_parsed_intent(
            user_id=user_id, message=message,
            parsed_intent=parsed_intent,
            agent_type="llm_personalized_agent",
            use_agent_loop=True,
        )

    def recommend_for_user_with_parsed_intent(
        self,
        user_id: int,
        message: str,
        parsed_intent: dict,
        agent_type: str = "llm_personalized_agent",
        use_agent_loop: bool = True,
    ) -> dict:
        """个性化推荐核心方法。"""
        parsed_intent = self.normalize_parsed_intent(message, parsed_intent)

        # 获取用户画像
        if self.user_profile_service is None:
            return {
                "user_id": user_id, "message": message,
                "error": "user_profile_service is not available",
                "results": [],
            }

        profile = self.user_profile_service.get_profile(user_id)
        if profile is None:
            return {
                "user_id": user_id, "message": message,
                "error": "User profile not found.",
                "results": [],
            }

        seen_movie_ids = set()
        if self.user_seen_movies_service is not None:
            seen_movie_ids = self.user_seen_movies_service.get_seen_movie_ids(user_id)

        preferred_genres_text = profile.get("preferred_genres") or ""
        preferred_genres = [
            genre.strip()
            for genre in preferred_genres_text.split(",")
            if genre.strip()
        ]

        requested_genres = self.parse_genre_list(parsed_intent.get("genre"))
        if requested_genres:
            search_genres = [parsed_intent.get("genre")]
        else:
            search_genres = preferred_genres[:3] if preferred_genres else [None]

        enhanced_query = parsed_intent["query"]
        if preferred_genres and not requested_genres:
            enhanced_query = (
                parsed_intent["query"]
                + ". User historical preferred genres: "
                + ", ".join(preferred_genres)
            )

        top_k = parsed_intent["top_k"]
        merged = {}
        attempts = []

        # 搜索循环（保持确定性多轮逻辑，因为个性化场景更复杂）
        strategies = [
            ("personalized_strict", parsed_intent.get("min_vote_average"), parsed_intent.get("min_vote_count")),
            ("personalized_relax_quality", None, None),
        ]

        for strategy, min_va, min_vc in strategies:
            if len(merged) >= top_k:
                break
            for genre in search_genres:
                search_result = self.retrieval_service.search_movies(
                    query=enhanced_query, top_k=200,
                    genre=genre,
                    exclude_genre=parsed_intent.get("exclude_genre"),
                    year_from=parsed_intent.get("year_from"),
                    year_to=parsed_intent.get("year_to"),
                    min_vote_average=min_va,
                    min_vote_count=min_vc,
                )
                attempts.append({
                    "strategy": strategy, "genre": genre,
                    "min_vote_average": min_va, "min_vote_count": min_vc,
                    "result_count": len(search_result["results"]),
                })
                self._add_personalized_candidates(
                    merged, search_result, genre, strategy,
                    preferred_genres, requested_genres, seen_movie_ids, parsed_intent,
                )

        # 第三轮：放宽多类型
        if len(merged) < top_k and len(requested_genres) >= 2:
            for genre in requested_genres:
                search_result = self.retrieval_service.search_movies(
                    query=enhanced_query, top_k=200,
                    genre=genre,
                    exclude_genre=parsed_intent.get("exclude_genre"),
                    year_from=parsed_intent.get("year_from"),
                    year_to=parsed_intent.get("year_to"),
                    min_vote_average=6.0, min_vote_count=100,
                )
                attempts.append({
                    "strategy": "personalized_relax_multi_genre", "genre": genre,
                    "result_count": len(search_result["results"]),
                })
                self._add_personalized_candidates(
                    merged, search_result, genre, "personalized_relax_multi_genre",
                    preferred_genres, requested_genres, seen_movie_ids, parsed_intent,
                )

        # 第四轮：宽泛兜底
        if len(merged) < top_k:
            search_result = self.retrieval_service.search_movies(
                query=enhanced_query, top_k=200,
                genre=None,
                exclude_genre=parsed_intent.get("exclude_genre"),
                year_from=parsed_intent.get("year_from"),
                year_to=parsed_intent.get("year_to"),
                min_vote_average=None, min_vote_count=30,
            )
            attempts.append({
                "strategy": "personalized_broad_fallback", "genre": None,
                "result_count": len(search_result["results"]),
            })
            self._add_personalized_candidates(
                merged, search_result, None, "personalized_broad_fallback",
                preferred_genres, requested_genres, seen_movie_ids, parsed_intent,
            )

        results = list(merged.values())

        # 排序
        results = self._rank_personalized_candidates(results, requested_genres)

        # 质量分层
        final_results = self._quality_filter(results, top_k)

        # RAG 生成
        try:
            agent_response, individual_reasons = self.llm_service.generate_recommendation_response(
                user_message=message,
                parsed_intent=parsed_intent,
                candidates=final_results,
                user_profile=profile,
                attempts=attempts,
                individual_reasons=True,
            )
            rag_generated = True
            for movie, reason in zip(final_results, individual_reasons):
                movie["agent_reason"] = reason
        except Exception as e:
            logger.warning(f"RAG generation failed for user {user_id}: {e}")
            rag_generated = False
            agent_response = self._template_personalized_response(
                parsed_intent, profile, seen_movie_ids, final_results, top_k, attempts, requested_genres,
            )
            for movie in final_results:
                movie["agent_reason"] = (
                    f"根据你的偏好推荐{movie.get('title', 'Unknown')}"
                )

        # 添加质量标签
        for movie in final_results:
            quality_bucket = self._get_quality_bucket(movie)
            movie["quality_bucket"] = quality_bucket
            confidence_map = {3: "high", 2: "medium", 1: "low", 0: "very_low"}
            level_map = {3: "strong_match", 2: "acceptable_match", 1: "weak_fallback", 0: "low_quality_fallback"}
            movie["recommendation_confidence"] = confidence_map.get(quality_bucket, "very_low")
            movie["recommendation_level"] = level_map.get(quality_bucket, "low_quality_fallback")
            movie["is_low_quality_fallback"] = quality_bucket == 0
            movie["is_weak_fallback"] = quality_bucket <= 1

        fallback_used = any(a["strategy"] != "personalized_strict" for a in attempts)

        return {
            "user_id": user_id, "message": message,
            "agent_type": agent_type,
            "rag_generated": rag_generated,
            "agent_response": agent_response,
            "parsed_intent": parsed_intent,
            "profile": profile,
            "used_preferred_genres": preferred_genres[:3],
            "attempts": attempts,
            "fallback_used": fallback_used,
            "exclude_seen": True,
            "seen_movie_count": len(seen_movie_ids),
            "candidate_count": len(results),
            "requested_count": top_k,
            "returned_count": len(final_results),
            "results": final_results,
        }

    # ──────────────────── 个性化辅助方法 ────────────────────

    def _add_personalized_candidates(
        self, merged, search_result, matched_genre, strategy,
        preferred_genres, requested_genres, seen_movie_ids, parsed_intent,
    ):
        """向合并集中添加个性化候选（含过滤和评分）。"""
        for movie in search_result["results"]:
            movie_id = movie["id"]
            if movie_id in seen_movie_ids:
                continue

            if requested_genres:
                movie_genres_lower = (movie.get("genres") or "").lower()
                if not any(
                    g.lower() in movie_genres_lower for g in requested_genres
                ):
                    continue

            movie_genres_lower = (movie.get("genres") or "").lower()
            matched_requested = self.get_matched_required_genres(
                movie.get("genres"), requested_genres,
            )

            user_genre_boost = 0.0
            for rank, pg in enumerate(preferred_genres[:5]):
                if pg.lower() in movie_genres_lower:
                    user_genre_boost = max(user_genre_boost, max(0.12 - rank * 0.02, 0.02))
            if requested_genres:
                user_genre_boost = min(user_genre_boost * 0.3, 0.06)

            request_genre_bonus = 0.0
            if requested_genres:
                if len(matched_requested) == len(requested_genres):
                    request_genre_bonus = 0.30
                elif len(matched_requested) > 0:
                    request_genre_bonus = 0.03
                else:
                    request_genre_bonus = -0.30

            strategy_bonus_map = {
                "personalized_strict": 0.10,
                "personalized_relax_quality": 0.06,
                "personalized_relax_multi_genre": -0.05,
                "personalized_broad_fallback": -0.15,
            }
            strategy_bonus = strategy_bonus_map.get(strategy, 0.0)

            agent_adjusted_score = (
                movie["final_score"] + request_genre_bonus + user_genre_boost + strategy_bonus
            )

            movie["agent_strategy"] = strategy
            movie["matched_user_genre"] = matched_genre
            movie["matched_requested_genres"] = matched_requested
            movie["requested_genre_match_count"] = len(matched_requested)
            movie["requested_genre_total_count"] = len(requested_genres)
            movie["user_genre_boost"] = round(user_genre_boost, 4)
            movie["request_genre_bonus"] = round(request_genre_bonus, 4)
            movie["strategy_bonus"] = round(strategy_bonus, 4)
            movie["agent_adjusted_score"] = round(agent_adjusted_score, 4)

            if movie_id not in merged:
                merged[movie_id] = movie
            elif agent_adjusted_score > merged[movie_id].get("agent_adjusted_score", 0):
                merged[movie_id] = movie

    def _get_quality_bucket(self, movie: dict) -> int:
        vote_average = float(movie.get("vote_average") or 0)
        vote_count = int(movie.get("vote_count") or 0)
        ml_rating_mean = float(movie.get("ml_rating_mean") or 0)
        ml_rating_count = int(movie.get("ml_rating_count") or 0)

        if (vote_average >= 6.8 or ml_rating_mean >= 4.0) and (vote_count >= 500 or ml_rating_count >= 400):
            return 3
        if (vote_average >= 6.0 or ml_rating_mean >= 3.5) and (vote_count >= 100 or ml_rating_count >= 80):
            return 2
        if (vote_average >= 4.0 or ml_rating_mean >= 2.0) and (vote_count >= 50 or ml_rating_count >= 30):
            return 1
        return 0

    def _rank_personalized_candidates(self, results, requested_genres):
        def is_full_match(movie):
            if not requested_genres:
                return True
            return movie.get("requested_genre_match_count", 0) >= len(requested_genres)

        def get_score(movie):
            return float(movie.get("agent_adjusted_score") or movie.get("final_score") or 0)

        strategy_priority = {
            "personalized_strict": 5,
            "personalized_relax_quality": 4,
            "personalized_relax_multi_genre": 3,
            "personalized_genre_preserving_fallback": 2,
            "personalized_broad_fallback": 1,
        }

        results.sort(
            key=lambda m: (
                1 if is_full_match(m) else 0,
                self._get_quality_bucket(m),
                strategy_priority.get(m.get("agent_strategy"), 0),
                get_score(m),
                m.get("quality_score") or 0,
                m.get("ml_rating_mean") or 0,
                m.get("ml_rating_count") or 0,
            ),
            reverse=True,
        )
        return results

    def _quality_filter(self, results, top_k):
        acceptable = [m for m in results if self._get_quality_bucket(m) >= 1]
        low_quality = [m for m in results if self._get_quality_bucket(m) == 0]
        if len(acceptable) >= top_k:
            return acceptable[:top_k]
        return (acceptable + low_quality)[:top_k]

    # ──────────────────── 模板回退（LLM 不可用时的保底） ────────────────────

    def _template_response(self, parsed_intent, attempts, final_results):
        """LLM 不可用时的模板化回复。"""
        if not final_results:
            return "没有找到符合条件的电影。尝试放宽筛选条件。"

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

        titles = [m["title"] for m in final_results]
        strict_count = attempts[0].get("result_count", 0) if attempts else 0
        fallback_text = (
            f"严格条件下找到 {strict_count} 部电影，系统自动放宽了条件。"
            if len(attempts) > 1
            else "系统使用严格条件找到了足够的结果。"
        )

        return (
            f"[模板回退 - LLM 不可用] 筛选条件：{filter_text}。"
            f"{fallback_text}"
            f"推荐：{'、'.join(titles)}。"
            "排序综合考虑了语义相关性、类型匹配、TMDB 质量分和 MovieLens 评分统计。"
        )

    def _template_reason(self, movie, parsed_intent):
        """LLM 不可用时的模板化单条理由。"""
        parts = []
        matched = movie.get("matched_required_genres") or []
        if matched:
            parts.append("匹配类型: " + ", ".join(matched))
        strategy = movie.get("agent_strategy", "")
        strategy_labels = {
            "strict": "严格匹配",
            "relax_quality_filters": "放宽质量门槛后匹配",
            "relax_multi_genre_to_primary_genre": "放宽多类型要求后匹配",
            "broad_semantic_fallback": "宽泛语义兜底",
            "llm_modified": "LLM 调整参数后匹配",
        }
        if strategy in strategy_labels:
            parts.append(strategy_labels[strategy])
        return "; ".join(parts) if parts else "根据你的需求推荐"

    def _template_personalized_response(
        self, parsed_intent, profile, seen_movie_ids, final_results, top_k, attempts, requested_genres,
    ):
        """LLM 不可用时的个性化模板回复。"""
        if not final_results:
            return "没有为用户找到合适的新电影。"

        titles = [m["title"] for m in final_results]
        preferred = profile.get("preferred_genres", "")
        strict_count = sum(
            a.get("result_count", 0) for a in attempts
            if a.get("strategy") == "personalized_strict"
        )
        fallback_text = (
            f"严格条件下找到 {strict_count} 部候选，使用 fallback 补充。"
            if len([a for a in attempts if a["strategy"] != "personalized_strict"]) > 0
            else "来自严格筛选条件。"
        )

        return (
            f"[模板回退 - LLM 不可用] 用户偏好类型: {preferred}。"
            f"已排除 {len(seen_movie_ids)} 部已看电影。"
            f"{fallback_text}"
            f"推荐：{'、'.join(titles)}。"
        )
