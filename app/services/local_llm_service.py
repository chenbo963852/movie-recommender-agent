import json
import re
import torch

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLLMService:
    def __init__(self, model_path: str = "local_models/Qwen2.5-0.5B-Instruct"):
        self.model_path = Path(model_path)
        self.tokenizer = None
        self.model = None

    def load_model(self):
        print(f"Loading local LLM from: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )

        print("Local LLM loaded.")

    def chat(self, user_message: str, system_message: str | None = None) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Local LLM is not loaded.")

        if system_message is None:
            system_message = "你是一个电影推荐系统的参数抽取助手。请尽量输出简洁、结构化的结果。"

        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return response.strip()

    def extract_search_params(self, user_prompt: str) -> dict:
        system_message = """
你是一个电影推荐系统的参数抽取器。

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
12. “不要恐怖片” 表示 exclude_genre = "Horror"
13. “2010年之后” 表示 year_from = 2010
14. 如果用户说推荐5部，top_k = 5
"""

        raw_response = self.chat(
            user_message=user_prompt,
            system_message=system_message
        )

        params = self._parse_json_from_text(raw_response)

        if params is None:
            params = self._fallback_extract_params(user_prompt)

        params = self._patch_params_by_rules(user_prompt, params)

        return params

    def _parse_json_from_text(self, text: str) -> dict | None:
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
            params["min_vote_count"] = 500

        if "热门" in user_prompt:
            params["min_vote_count"] = 500

        if "科幻" in user_prompt and "惊悚" in user_prompt:
            params["query"] = "science fiction thriller"
        elif "科幻" in user_prompt:
            params["query"] = "science fiction"
        elif "惊悚" in user_prompt:
            params["query"] = "thriller"

        params.setdefault("query", user_prompt)
        params.setdefault("genre", None)
        params.setdefault("exclude_genre", None)
        params.setdefault("year_from", None)
        params.setdefault("year_to", None)
        params.setdefault("min_vote_average", None)
        params.setdefault("min_vote_count", None)
        params.setdefault("top_k", 5)

        return params
