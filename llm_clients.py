from __future__ import annotations

import json
import math
import random
from abc import ABC, abstractmethod
from typing import Any

from .config import ASSET_TICKERS, EXPERIMENT_CONFIG
from .lmstudio_openai_chat import LMStudioOpenAIChat
from tradingagents_tsfm_modified_v5.tradingagents.llms.local_qwen import LocalQwenChat


class BaseLLMClient(ABC):
    """LLM 客户端统一接口。"""

    @abstractmethod
    def invoke(self, messages: Any) -> Any:
        """调用模型并返回原始响应对象。"""

    def inspect_messages(self, messages: Any) -> dict[str, Any]:
        token_count, source = self._estimate_input_tokens(messages)
        budget = getattr(self, "input_token_budget", None)
        return {
            "input_token_count": token_count,
            "input_token_count_source": source,
            "input_token_budget": budget,
            "input_token_over_budget": (
                budget is not None and token_count is not None and token_count > budget
            ),
        }

    @abstractmethod
    def _estimate_input_tokens(self, messages: Any) -> tuple[int | None, str]:
        """估算或精确计算输入 token 数。"""


def _approximate_input_tokens(messages: Any) -> tuple[int, str]:
    # LM Studio/OpenAI 兼容路径拿不到底层 tokenizer 时，退化成字符近似。
    payload = json.dumps(messages, ensure_ascii=False, default=str)
    return max(1, math.ceil(len(payload) / 4)), "approx_chars_div4"


class MockLLMClient(BaseLLMClient):
    """用于无卡调试的 mock LLM。"""

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name

    def invoke(self, messages: Any) -> Any:
        # 使用外部已设置好的随机种子。
        weights = {t: random.random() for t in ASSET_TICKERS}
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        response = f'''```json
{{
  "action": "rebalance",
  "weights": {{{", ".join(f'"{k}": {v:.4f}' for k, v in weights.items())}}},
  "confidence": 0.7,
  "reasoning": "Mock decision for debugging"
}}
```'''

        class MockResponse:
            def __init__(self, content: str):
                self.content = content

        return MockResponse(response)

    def _estimate_input_tokens(self, messages: Any) -> tuple[int | None, str]:
        return _approximate_input_tokens(messages)


class QwenLLMClient(BaseLLMClient):
    """本地 Qwen 客户端适配器。"""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        input_token_budget: int | None = None,
    ):
        self._client = LocalQwenChat(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        self.input_token_budget = input_token_budget

    def invoke(self, messages: Any) -> Any:
        return self._client.invoke(messages)

    def _estimate_input_tokens(self, messages: Any) -> tuple[int | None, str]:
        if hasattr(self._client, "count_input_tokens"):
            return self._client.count_input_tokens(messages), "qwen_tokenizer"
        return _approximate_input_tokens(messages)


class LMStudioLLMClient(BaseLLMClient):
    """LM Studio OpenAI 兼容客户端适配器。"""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        input_token_budget: int | None = None,
    ):
        self._client = LMStudioOpenAIChat(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        self.input_token_budget = input_token_budget

    def invoke(self, messages: Any) -> Any:
        return self._client.invoke(messages)

    def _estimate_input_tokens(self, messages: Any) -> tuple[int | None, str]:
        return _approximate_input_tokens(messages)


def build_llm_client(debug: bool, use_mock_llm: bool = False) -> BaseLLMClient:
    """根据配置构建统一的 LLM 客户端。"""
    llm_model = (
        EXPERIMENT_CONFIG["debug_llm"]
        if debug
        else EXPERIMENT_CONFIG["production_llm"]
    )
    llm_provider = EXPERIMENT_CONFIG.get("llm_provider", "qwen")

    if use_mock_llm:
        print("Using MockLLM for debugging...")
        return MockLLMClient(model_name="mock")

    if llm_provider == "qwen":
        return QwenLLMClient(
            model_name=llm_model,
            temperature=0.0,
            max_new_tokens=int(EXPERIMENT_CONFIG.get("llm_max_new_tokens") or 1024),
            input_token_budget=EXPERIMENT_CONFIG.get("llm_input_token_budget"),
        )

    return LMStudioLLMClient(
        model_name=llm_model,
        base_url=EXPERIMENT_CONFIG["lmstudio_base_url"],
        api_key=EXPERIMENT_CONFIG.get("lmstudio_api_key"),
        temperature=0.0,
        max_new_tokens=int(EXPERIMENT_CONFIG.get("llm_max_new_tokens") or 1024),
        input_token_budget=EXPERIMENT_CONFIG.get("llm_input_token_budget"),
    )
