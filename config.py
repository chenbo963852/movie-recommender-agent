"""
集中管理 LLM 配置，从环境变量 / .env 文件读取。

使用方式:
    from config import llm_config
    service = LLMService(backend=llm_config.backend, config=llm_config.to_dict())
"""

import os
from pathlib import Path

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


class LLMConfig:
    def __init__(self):
        self.backend: str = os.getenv("LLM_BACKEND", "local")
        self.api_key: str = os.getenv("LLM_API_KEY", "")
        self.base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
        # 本地模型路径（backend=local 时生效）
        self.local_model_path: str = os.getenv(
            "LLM_LOCAL_MODEL_PATH",
            str(Path(__file__).parent / "local_models" / "Qwen2.5-0.5B-Instruct"),
        )

    def to_dict(self) -> dict:
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "local_model_path": self.local_model_path,
        }

    @property
    def is_cloud(self) -> bool:
        return self.backend == "cloud"

    @property
    def is_local(self) -> bool:
        return self.backend == "local"


llm_config = LLMConfig()
