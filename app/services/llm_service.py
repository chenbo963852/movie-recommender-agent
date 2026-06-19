"""
统一的 LLM 服务层。

支持两种后端：
- cloud: OpenAI 兼容 API（OpenAI / Qwen API / DeepSeek / vLLM 等）
- local: 本地 transformers 模型（Qwen2.5-0.5B-Instruct）

所有 Agent 和 RAG 相关的 LLM 调用都通过这一层，方便切换后端。
"""

import json
import re
import logging

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self, backend: str = "cloud", config: dict | None = None):
        """
        Args:
            backend: "cloud" | "local"
            config: backend 相关配置，包含 api_key, base_url, model 等
        """
        self.backend = backend
        self.config = config or {}
        self._client = None
        self._local_model = None
        self._local_tokenizer = None

        if self.backend == "cloud":
            self._init_cloud_client()

    # ──────────────────── 初始化 ────────────────────

    def _init_cloud_client(self):
        """初始化 OpenAI 兼容客户端。"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for cloud backend. "
                "Install it with: pip install openai"
            )

        api_key = self.config.get("api_key", "")
        base_url = self.config.get("base_url", "https://api.openai.com/v1")

        if not api_key:
            raise ValueError(
                "LLM_API_KEY is required for cloud backend. "
                "Set it in .env or environment variables."
            )

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = self.config.get("model", "gpt-4o-mini")
        logger.info(f"Cloud LLM client initialized: model={self._model}, base_url={base_url}")

    def _init_local_model(self):
        """延迟加载本地模型（只在首次使用时加载）。"""
        if self._local_model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from pathlib import Path

        model_path = Path(self.config.get(
            "local_model_path",
            "local_models/Qwen2.5-0.5B-Instruct",
        ))

        logger.info(f"Loading local LLM from: {model_path}")

        self._local_tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self._local_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("Local LLM loaded.")

    # ──────────────────── 统一 chat 接口 ────────────────────

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        """
        统一的聊天接口。

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
            temperature: 生成温度
            max_tokens: 最大输出 token 数

        Returns:
            LLM 的文本响应
        """
        if self.backend == "cloud":
            return self._chat_cloud(messages, temperature, max_tokens)
        else:
            return self._chat_local(messages, temperature, max_tokens)

    def _chat_cloud(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _chat_local(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        import torch

        self._init_local_model()

        text = self._local_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        model_inputs = self._local_tokenizer(
            [text], return_tensors="pt",
        ).to(self._local_model.device)

        with torch.no_grad():
            generated_ids = self._local_model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self._local_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True,
        )[0]

        return response.strip()

    # ──────────────────── 意图提取 ────────────────────

    def extract_search_params(self, user_prompt: str) -> dict:
        """
        从自然语言中提取结构化搜索参数。

        优先使用 LLM，失败时回退到规则解析。
        """
        # 尝试 LLM 提取
        try:
            return self._extract_with_llm(user_prompt)
        except Exception as e:
            logger.warning(f"LLM intent extraction failed: {e}, falling back to rules")
            return self._fallback_extract_params(user_prompt)

    def _extract_with_llm(self, user_prompt: str) -> dict:
        system_message = """你是一个电影推荐系统的参数抽取器。

你只能输出 JSON，不要输出解释，不要输出电影名。

JSON 字段必须是：
{
  "query": string,
  "genre": string or null,
  "exclude_genre": string or null,
  "year_from": number or null,
  "year_to": number or null,
  "min_vote_average": number or null,
  "min_vote_count": number or null,
  "top_k": number
}

规则：
1. 科幻 = Science Fiction
2. 惊悚 = Thriller
3. 恐怖 = Horror
4. 喜剧 = Comedy
5. 爱情 = Romance
6. 动作 = Action
7. 剧情 = Drama
8. 动画 = Animation
9. 犯罪 = Crime
10. 高评分默认 min_vote_average = 7.0
11. 热门默认 min_vote_count = 500
12. "不要恐怖片" 表示 exclude_genre = "Horror"
13. "2010年之后" 表示 year_from = 2010
14. 如果用户说推荐5部，top_k = 5"""

        raw_response = self.chat([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ])

        params = self._parse_json_from_text(raw_response)
        if params is None:
            params = self._fallback_extract_params(user_prompt)

        params = self._patch_params_by_rules(user_prompt, params)
        return params

    # ──────────────────── Agent 决策 ────────────────────

    def decide_next_action(self, context: str) -> dict:
        """
        让 LLM 根据当前搜索状态决定下一步动作。

        Args:
            context: 包含用户需求、当前候选数、已尝试策略的文本

        Returns:
            {
                "action": "search_movies" | "finalize",
                "reasoning": "简短推理",
                "params": { ... }   # 仅当 action=search_movies 时
            }
        """
        system_message = """你是一个电影推荐 Agent 的决策模块。

你的任务是根据当前搜索结果，决定下一步：继续搜索（可能调整参数），还是结束搜索进入推荐生成阶段。

规则：
- 如果候选电影 >= 用户要求的数量，且质量尚可 → finalize
- 如果候选不足 → 适当放宽条件再搜索（降低评分门槛、放宽年份、减少类型要求等）
- 如果已经尝试多种方式依然不够 → finalize（用现有候选生成推荐，诚实告知用户）
- 最多尝试 4-5 轮

输出格式（严格 JSON）：
{
  "action": "search_movies",
  "reasoning": "当前只有2部候选，不足5部要求，放宽评分门槛再搜",
  "params": {
    "query": "语义搜索文本",
    "genre": "Science Fiction,Thriller",
    "exclude_genre": "Horror",
    "year_from": 2010,
    "year_to": null,
    "min_vote_average": 6.0,
    "min_vote_count": 150
  }
}

或

{
  "action": "finalize",
  "reasoning": "候选集已达到12部，质量尚可，可以生成推荐"
}"""

        raw_response = self.chat([
            {"role": "system", "content": system_message},
            {"role": "user", "content": context},
        ])

        decision = self._parse_json_from_text(raw_response)

        if decision is None or "action" not in decision:
            # 解析失败，返回默认：finalize
            logger.warning(f"Agent decision parsing failed, defaulting to finalize. Raw: {raw_response[:200]}")
            return {"action": "finalize", "reasoning": "decision parsing failed, defaulting to finalize"}

        return decision

    # ──────────────────── RAG 生成 ────────────────────

    def generate_recommendation_response(
        self,
        user_message: str,
        parsed_intent: dict,
        candidates: list[dict],
        user_profile: dict | None = None,
        attempts: list[dict] | None = None,
        individual_reasons: bool = False,
    ) -> str | list[str]:
        """
        RAG 核心方法 —— 基于检索到的电影元数据生成推荐回复。

        Args:
            user_message: 用户原始输入
            parsed_intent: 解析出的筛选条件
            candidates: top-N 候选电影，每部包含 title, genres, year,
                        vote_average, vote_count, overview, directors, cast, keywords 等
            user_profile: 可选的用户画像
            attempts: 搜索尝试记录（用于说明回退情况）
            individual_reasons: 若为 True，返回每部电影的单独推荐理由列表

        Returns:
            自然语言推荐回复文本，或 individual_reasons=True 时返回 list[str]
        """
        if not candidates:
            return "抱歉，没有找到符合你条件的电影。建议放宽筛选条件再试试。"

        # 构建候选电影摘要
        movie_summaries = []
        for i, movie in enumerate(candidates, 1):
            parts = [f"{i}. **{movie.get('title', 'Unknown')}**"]

            genres = movie.get("genres", "")
            if genres:
                parts.append(f"   - 类型: {genres}")

            year = movie.get("year")
            if year:
                parts.append(f"   - 年份: {year}")

            vote_avg = movie.get("vote_average")
            vote_count = movie.get("vote_count")
            if vote_avg is not None and vote_count is not None:
                try:
                    parts.append(f"   - TMDB 评分: {float(vote_avg):.1f}/10 ({int(vote_count)} 人)")
                except (ValueError, TypeError):
                    pass

            ml_mean = movie.get("ml_rating_mean")
            ml_count = movie.get("ml_rating_count")
            if ml_mean is not None and ml_count is not None:
                try:
                    parts.append(f"   - MovieLens 评分: {float(ml_mean):.2f}/5 ({int(ml_count)} 人)")
                except (ValueError, TypeError):
                    pass

            directors = movie.get("directors", "")
            if directors:
                parts.append(f"   - 导演: {directors}")

            cast = movie.get("cast", "")
            if cast:
                parts.append(f"   - 主演: {cast}")

            overview = movie.get("overview", "")
            if overview:
                # 截断过长的概述
                overview_short = overview[:200] + "..." if len(str(overview)) > 200 else str(overview)
                parts.append(f"   - 简介: {overview_short}")

            # Agent 策略信息
            strategy = movie.get("agent_strategy", "")
            if strategy:
                strategy_labels = {
                    "strict": "严格匹配",
                    "relax_quality_filters": "放宽质量门槛",
                    "relax_multi_genre_to_primary_genre": "放宽多类型要求",
                    "broad_semantic_fallback": "宽泛语义兜底",
                    "personalized_strict": "个性化严格匹配",
                    "personalized_relax_quality": "个性化放宽质量",
                    "personalized_relax_multi_genre": "个性化放宽类型",
                    "personalized_broad_fallback": "个性化宽泛兜底",
                }
                label = strategy_labels.get(strategy, strategy)
                parts.append(f"   - 召回策略: {label}")

            movie_summaries.append("\n".join(parts))

        # 构建筛选条件说明
        filter_parts = []
        if parsed_intent.get("genre"):
            filter_parts.append(f"类型: {parsed_intent['genre']}")
        if parsed_intent.get("exclude_genre"):
            filter_parts.append(f"排除: {parsed_intent['exclude_genre']}")
        if parsed_intent.get("year_from"):
            filter_parts.append(f"{parsed_intent['year_from']} 年之后")
        if parsed_intent.get("year_to"):
            filter_parts.append(f"{parsed_intent['year_to']} 年之前")
        if parsed_intent.get("min_vote_average"):
            filter_parts.append(f"评分 >= {parsed_intent['min_vote_average']}")
        if parsed_intent.get("min_vote_count"):
            filter_parts.append(f"评分人数 >= {parsed_intent['min_vote_count']}")
        filter_text = "、".join(filter_parts) if filter_parts else "无特殊筛选条件"

        # 构建回退说明
        fallback_note = ""
        if attempts:
            strict_count = attempts[0].get("result_count", 0) if attempts else 0
            total_attempts = len(attempts)
            if total_attempts > 1:
                fallback_note = (
                    f"\n注意：严格条件下仅找到 {strict_count} 部候选，"
                    f"系统经过 {total_attempts} 轮搜索策略调整后获得以上结果。"
                )
            else:
                fallback_note = "\n以上结果均来自严格条件匹配。"

        # 用户画像提示
        profile_note = ""
        if user_profile:
            preferred = user_profile.get("preferred_genres", "")
            if preferred:
                profile_note = f"\n该用户偏好类型: {preferred}。"

        system_message = """你是一个专业的电影推荐助手。

你的回复应该：
1. 自然、友好、有信息量
2. 为每部推荐电影写一句简短的推荐理由，要基于提供的**真实元数据**（类型、导演、评分、剧情等）
3. 不要编造电影信息，只使用提供的数据
4. 如果使用了回退策略（放宽条件），诚实说明
5. 用中文回复"""

        user_context = f"""用户需求: {user_message}

解析的筛选条件: {filter_text}
{profile_note}
{fallback_note}

从数据库中检索到的候选电影（共 {len(candidates)} 部）:

{chr(10).join(movie_summaries)}

请基于以上检索到的真实电影数据，为用户生成推荐回复。"""

        generated = self.chat([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_context},
        ], temperature=0.3, max_tokens=1024)

        # 如果需要单独生成每部电影的推荐理由
        if individual_reasons and len(candidates) > 0:
            reasons = self._generate_individual_reasons(candidates, user_message)
            # 返回整体回复和单条理由
            return generated, reasons

        return generated

    def _generate_individual_reasons(
        self,
        candidates: list[dict],
        user_message: str,
    ) -> list[str]:
        """为每部候选电影生成单独的一句推荐理由。"""
        reasons = []
        for movie in candidates:
            title = movie.get("title", "Unknown")
            genres = movie.get("genres", "")
            year = movie.get("year", "")
            directors = movie.get("directors", "")
            overview = movie.get("overview", "")
            overview_short = str(overview)[:150] if overview else ""

            prompt = (
                f"用户想找: {user_message}\n"
                f"电影: {title} ({year}), 类型: {genres}"
            )
            if directors:
                prompt += f", 导演: {directors}"
            if overview_short:
                prompt += f", 简介: {overview_short}"
            prompt += "\n请用一句话中文推荐理由（不超过50字）:"

            try:
                reason = self.chat([
                    {"role": "system", "content": "你是一个电影推荐助手。用一句话写推荐理由，只写理由本身，不要前缀。"},
                    {"role": "user", "content": prompt},
                ], temperature=0.2, max_tokens=80)
                reasons.append(reason.strip())
            except Exception as e:
                logger.warning(f"Failed to generate reason for {title}: {e}")
                reasons.append(f"根据你的偏好推荐{title}")

        return reasons

    # ──────────────────── 工具函数 ────────────────────

    def _parse_json_from_text(self, text: str) -> dict | None:
        """从文本中提取 JSON 对象。"""
        if not text:
            return None

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def _fallback_extract_params(self, user_prompt: str) -> dict:
        """LLM 解析失败时的规则回退。"""
        prompt = user_prompt.lower()

        genre_list = []
        if "科幻" in user_prompt:
            genre_list.append("Science Fiction")
        if "惊悚" in user_prompt:
            genre_list.append("Thriller")
        if "喜剧" in user_prompt:
            genre_list.append("Comedy")
        if "爱情" in user_prompt:
            genre_list.append("Romance")
        if "动作" in user_prompt:
            genre_list.append("Action")
        if "剧情" in user_prompt:
            genre_list.append("Drama")
        if "动画" in user_prompt:
            genre_list.append("Animation")
        if "犯罪" in user_prompt:
            genre_list.append("Crime")

        exclude_genre = None
        if "不要恐怖" in user_prompt or "非恐怖" in user_prompt:
            exclude_genre = "Horror"

        year_from = None
        year_match = re.search(r"(\d{4})年之后", user_prompt)
        if year_match:
            year_from = int(year_match.group(1))

        top_k = 5
        top_k_match = re.search(r"推荐\s*(\d+)\s*部", user_prompt)
        if top_k_match:
            top_k = int(top_k_match.group(1))

        min_vote_average = None
        if "高评分" in user_prompt or "高分" in user_prompt:
            min_vote_average = 7.0

        min_vote_count = None
        if "热门" in user_prompt or "高评分" in user_prompt:
            min_vote_count = 500

        return {
            "query": user_prompt,
            "genre": ",".join(genre_list) if genre_list else None,
            "exclude_genre": exclude_genre,
            "year_from": year_from,
            "year_to": None,
            "min_vote_average": min_vote_average,
            "min_vote_count": min_vote_count,
            "top_k": top_k,
        }

    def _patch_params_by_rules(self, user_prompt: str, params: dict) -> dict:
        """用规则修正 LLM 的常见错误。"""
        genre_list = []

        if "科幻" in user_prompt:
            genre_list.append("Science Fiction")
        if "惊悚" in user_prompt:
            genre_list.append("Thriller")
        if "恐怖" in user_prompt and "不要恐怖" not in user_prompt:
            genre_list.append("Horror")
        if "喜剧" in user_prompt:
            genre_list.append("Comedy")
        if "爱情" in user_prompt:
            genre_list.append("Romance")
        if "动作" in user_prompt:
            genre_list.append("Action")
        if "剧情" in user_prompt:
            genre_list.append("Drama")
        if "动画" in user_prompt:
            genre_list.append("Animation")
        if "犯罪" in user_prompt:
            genre_list.append("Crime")

        if genre_list:
            params["genre"] = ",".join(genre_list)

        if (
            "不要恐怖" in user_prompt
            or "不看恐怖" in user_prompt
            or "排除恐怖" in user_prompt
            or "非恐怖" in user_prompt
        ):
            params["exclude_genre"] = "Horror"

        year_match = re.search(r"(\d{4})年之后", user_prompt)
        if year_match:
            params["year_from"] = int(year_match.group(1))

        top_k_match = re.search(r"推荐\s*(\d+)\s*部", user_prompt)
        if top_k_match:
            params["top_k"] = int(top_k_match.group(1))

        if "高评分" in user_prompt or "高分" in user_prompt:
            params["min_vote_average"] = 7.0
            params.setdefault("min_vote_count", 500)

        if "热门" in user_prompt:
            params["min_vote_count"] = 500

        # 确保必填字段存在
        params.setdefault("query", user_prompt)
        params.setdefault("genre", None)
        params.setdefault("exclude_genre", None)
        params.setdefault("year_from", None)
        params.setdefault("year_to", None)
        params.setdefault("min_vote_average", None)
        params.setdefault("min_vote_count", None)
        params.setdefault("top_k", 5)

        return params
