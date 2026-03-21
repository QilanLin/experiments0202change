from __future__ import annotations

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


class QwenLLMClient(BaseLLMClient):
    """本地 Qwen 客户端适配器。"""

    def __init__(self, model_name: str, temperature: float = 0.0):
        self._client = LocalQwenChat(
            model_name=model_name,
            temperature=temperature,
        )

    def invoke(self, messages: Any) -> Any:
        return self._client.invoke(messages)


class LMStudioLLMClient(BaseLLMClient):
    """LM Studio OpenAI 兼容客户端适配器。"""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.0,
    ):
        self._client = LMStudioOpenAIChat(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )

    def invoke(self, messages: Any) -> Any:
        return self._client.invoke(messages)


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
        return QwenLLMClient(model_name=llm_model, temperature=0.0)

    return LMStudioLLMClient(
        model_name=llm_model,
        base_url=EXPERIMENT_CONFIG["lmstudio_base_url"],
        api_key=EXPERIMENT_CONFIG.get("lmstudio_api_key"),
        temperature=0.0,
    )
