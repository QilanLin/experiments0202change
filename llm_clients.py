from __future__ import annotations

import json
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - optional dependency fallback
    AutoTokenizer = None  # type: ignore
try:
    from zai import ZhipuAiClient
except Exception:  # pragma: no cover - optional dependency fallback
    ZhipuAiClient = None  # type: ignore

from .config import ASSET_TICKERS, EXPERIMENT_CONFIG
from .lmstudio_openai_chat import LMStudioOpenAIChat
from tradingagents_tsfm_modified_v5.tradingagents.llms.local_qwen import LocalQwenChat

SUPPORTED_LLM_PROVIDERS = ("qwen", "lmstudio", "zhipu")


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


_HF_TOKEN_COUNTERS: dict[str, Any] = {}


def _normalize_messages_for_token_count(messages: Any) -> Any:
    if isinstance(messages, dict) and "messages" in messages:
        messages = messages["messages"]

    if not isinstance(messages, list):
        return str(messages)

    normalized = []
    for item in messages:
        if isinstance(item, dict):
            normalized.append(
                {
                    "role": str(item.get("role", "user")),
                    "content": str(item.get("content", "")),
                }
            )
        else:
            normalized.append({"role": "user", "content": str(item)})
    return normalized


def _try_build_hf_token_counter(model_name: str):
    if AutoTokenizer is None:
        return None

    if model_name in _HF_TOKEN_COUNTERS:
        return _HF_TOKEN_COUNTERS[model_name]

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:
        _HF_TOKEN_COUNTERS[model_name] = None
        return None

    def _counter(messages: Any) -> int | None:
        normalized = _normalize_messages_for_token_count(messages)
        try:
            if (
                isinstance(normalized, list)
                and hasattr(tokenizer, "apply_chat_template")
            ):
                prompt = tokenizer.apply_chat_template(
                    normalized,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            payload = json.dumps(normalized, ensure_ascii=False, default=str)
            return len(tokenizer(payload, add_special_tokens=False)["input_ids"])
        except Exception:
            return None

    _HF_TOKEN_COUNTERS[model_name] = _counter
    return _counter


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
        temperature: float,
        max_new_tokens: int,
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
        temperature: float,
        max_new_tokens: int,
        api_key: str | None = None,
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
        self._token_counter = _try_build_hf_token_counter(model_name)

    def invoke(self, messages: Any) -> Any:
        return self._client.invoke(messages)

    def _estimate_input_tokens(self, messages: Any) -> tuple[int | None, str]:
        if self._token_counter is not None:
            counted = self._token_counter(messages)
            if counted is not None:
                return counted, "hf_tokenizer"
        return _approximate_input_tokens(messages)


@dataclass
class _SimpleContentResponse:
    content: str


class ZhipuLLMClient(BaseLLMClient):
    """智谱 Z.ai Python SDK 客户端适配器。"""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_new_tokens: int,
        api_key: str | None = None,
        base_url: str | None = None,
        input_token_budget: int | None = None,
    ) -> None:
        if ZhipuAiClient is None:
            raise ImportError(
                "zai-sdk is required for llm_provider='zhipu'. "
                "Install with: pip install zai-sdk"
            )
        if not api_key:
            raise ValueError(
                "Missing Zhipu API key. Set env `ZAI_API_KEY` (or `ZHIPU_API_KEY`)."
            )
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.input_token_budget = input_token_budget
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url.rstrip("/") + "/"
        self._client = ZhipuAiClient(**client_kwargs)

    def invoke(self, messages: Any) -> Any:
        normalized = _normalize_messages_for_token_count(messages)
        if not isinstance(normalized, list):
            normalized = [{"role": "user", "content": str(normalized)}]
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=normalized,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        return _SimpleContentResponse(content=response.choices[0].message.content)

    def _estimate_input_tokens(self, messages: Any) -> tuple[int | None, str]:
        return _approximate_input_tokens(messages)


def _resolve_llm_runtime_settings() -> tuple[float, int, int | None]:
    temperature = float(EXPERIMENT_CONFIG.get("llm_temperature", 0.0))
    max_new_tokens = int(EXPERIMENT_CONFIG.get("llm_max_new_tokens") or 1024)
    input_token_budget = EXPERIMENT_CONFIG.get("llm_input_token_budget")
    if max_new_tokens <= 0:
        raise ValueError(f"Invalid llm_max_new_tokens={max_new_tokens}, expected > 0")
    if input_token_budget is not None and int(input_token_budget) <= 0:
        raise ValueError(
            f"Invalid llm_input_token_budget={input_token_budget}, expected > 0 or None"
        )
    return temperature, max_new_tokens, input_token_budget


def build_llm_client(debug: bool, use_mock_llm: bool = False) -> BaseLLMClient:
    """根据配置构建统一的 LLM 客户端。"""
    llm_provider = str(EXPERIMENT_CONFIG.get("llm_provider", "qwen")).strip().lower()
    if llm_provider == "zhipu":
        llm_model = (
            EXPERIMENT_CONFIG["zhipu_debug_llm"]
            if debug
            else EXPERIMENT_CONFIG["zhipu_production_llm"]
        )
    else:
        llm_model = (
            EXPERIMENT_CONFIG["debug_llm"]
            if debug
            else EXPERIMENT_CONFIG["production_llm"]
        )
    temperature, max_new_tokens, input_token_budget = _resolve_llm_runtime_settings()

    if use_mock_llm:
        print("Using MockLLM for debugging...")
        return MockLLMClient(model_name="mock")

    if llm_provider not in SUPPORTED_LLM_PROVIDERS:
        expected = ", ".join(SUPPORTED_LLM_PROVIDERS)
        raise ValueError(
            f"Unsupported llm_provider={llm_provider!r}. "
            f"Expected one of: {expected}"
        )

    print(
        "[LLM] provider={provider} model={model} temp={temp} max_new_tokens={max_new} "
        "input_token_budget={budget}".format(
            provider=llm_provider,
            model=llm_model,
            temp=temperature,
            max_new=max_new_tokens,
            budget=input_token_budget,
        ),
        flush=True,
    )

    if llm_provider == "qwen":
        return QwenLLMClient(
            model_name=llm_model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            input_token_budget=input_token_budget,
        )

    if llm_provider == "lmstudio":
        return LMStudioLLMClient(
            model_name=llm_model,
            base_url=EXPERIMENT_CONFIG["lmstudio_base_url"],
            api_key=EXPERIMENT_CONFIG.get("lmstudio_api_key"),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            input_token_budget=input_token_budget,
        )

    return ZhipuLLMClient(
        model_name=llm_model,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        api_key=EXPERIMENT_CONFIG.get("zhipu_api_key"),
        base_url=EXPERIMENT_CONFIG.get("zhipu_base_url"),
        input_token_budget=input_token_budget,
    )
