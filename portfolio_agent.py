"""
Portfolio Weight Allocation Agent

LLM决策输出：MAG7股票的权重分配
"""

from __future__ import annotations
from copy import deepcopy
from typing import Dict, Any, Optional

from .config import MAG7_TICKERS
from .decision_parser import DecisionParser
from .portfolio_models import PortfolioDecision, PortfolioState
from .prompt_builder import PromptBuilder


class PortfolioWeightAgent:
    """组合权重决策Agent"""
    
    SYSTEM_PROMPT = """You are a portfolio manager for MAG7 stocks (AAPL, GOOGL, AMZN, MSFT, META, TSLA, NVDA) plus CASH. You should trade daily.

Your task is to allocate portfolio weights based on the provided information.

RULES:
1. Output weights for all 8 assets (7 stocks + CASH) that sum to exactly 1.0
2. Weights must be between 0.0 and 1.0
3. You can set weight to 0 if you want to avoid a stock
4. CASH represents the uninvested portion (cash held in account)
5. Consider risk diversification
6. You can hold cash (CASH > 0) if you want to reduce exposure

OUTPUT FORMAT (MUST follow exactly):
```json
{
  "action": "rebalance" or "hold",
  "weights": {
    "AAPL": 0.XX,
    "GOOGL": 0.XX,
    "AMZN": 0.XX,
    "MSFT": 0.XX,
    "META": 0.XX,
    "TSLA": 0.XX,
    "NVDA": 0.XX,
    "CASH": 0.XX
  },
  "confidence": 0.X,
  "reasoning": "Brief explanation"
}
```

IMPORTANT: Weights MUST sum to 1.0 exactly. All 8 assets (7 stocks + CASH) must be included."""
    TOKEN_TRUNCATION_MARKER = "\n\n[TRUNCATED_FOR_TOKEN_BUDGET]"

    def __init__(self, llm, experiment_type: str, tsfm_format: Optional[int] = None):
        self.llm = llm
        self.experiment_type = experiment_type
        self.tsfm_format = tsfm_format
        self.prompt_builder = PromptBuilder()
        self.decision_parser = DecisionParser(
            experiment_type=experiment_type,
            tsfm_format=tsfm_format,
        )

    def prepare_request(
            self,
            current_date: str,
            fundamentals: Dict[str, str],
            price_history: Dict[str, Any],
            tsfm_forecasts: Optional[Dict[str, str]] = None,
            current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """构造给 LLM 的结构化输入。"""
        user_context = self.prompt_builder.build_context(
            current_date, fundamentals, price_history,
            tsfm_forecasts, current_weights
        )
        messages = self.prompt_builder.build_messages(self.SYSTEM_PROMPT, user_context)
        inspection = self._inspect_messages(messages)
        truncated = False
        truncation_strategy = None
        if inspection.get("input_token_over_budget"):
            budget = inspection.get("input_token_budget")
            if budget is not None:
                messages, inspection, truncated = self._truncate_user_message_to_fit_budget(
                    messages=messages,
                    budget=int(budget),
                )
                if truncated:
                    truncation_strategy = "user_tail_char_truncation"

        prompt_str = self.prompt_builder.format_prompt_for_saving(messages)
        return {
            "decision_date": current_date,
            "messages": messages,
            "prompt": prompt_str,
            "input_token_truncated": truncated,
            "input_token_truncation_strategy": truncation_strategy,
            **inspection,
        }

    def decide_from_request(self, prepared_request: Dict[str, Any]) -> PortfolioDecision:
        """使用已准备好的 messages 调 LLM，并解析输出。"""
        current_date = prepared_request["decision_date"]
        messages = prepared_request["messages"]
        prompt_str = prepared_request["prompt"]
        token_count = prepared_request.get("input_token_count")
        token_budget = prepared_request.get("input_token_budget")
        token_source = prepared_request.get("input_token_count_source", "unknown")

        if prepared_request.get("input_token_over_budget"):
            raise ValueError(
                f"LLM input token budget exceeded for {current_date}: "
                f"{token_count} > {token_budget} ({token_source})"
            )

        result = self.llm.invoke(messages)
        output = result.content if hasattr(result, 'content') else str(result)

        self.decision_parser.debug = getattr(self, "debug", False)
        return self.decision_parser.parse(output, current_date, prompt_str)

    def _inspect_messages(self, messages: list[dict[str, str]]) -> Dict[str, Any]:
        if hasattr(self.llm, "inspect_messages"):
            return self.llm.inspect_messages(messages)
        return {
            "input_token_count": None,
            "input_token_count_source": "unavailable",
            "input_token_budget": None,
            "input_token_over_budget": False,
        }

    def _truncate_user_message_to_fit_budget(
        self,
        *,
        messages: list[dict[str, str]],
        budget: int,
    ) -> tuple[list[dict[str, str]], Dict[str, Any], bool]:
        user_idx = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
        if user_idx is None:
            inspection = self._inspect_messages(messages)
            return messages, inspection, False

        user_content = str(messages[user_idx].get("content", ""))
        if not user_content:
            inspection = self._inspect_messages(messages)
            return messages, inspection, False

        marker = self.TOKEN_TRUNCATION_MARKER

        def _candidate_with_chars(chars_to_keep: int) -> tuple[list[dict[str, str]], Dict[str, Any]]:
            candidate = user_content[:chars_to_keep]
            if chars_to_keep < len(user_content):
                candidate += marker
            candidate_messages = deepcopy(messages)
            candidate_messages[user_idx]["content"] = candidate
            return candidate_messages, self._inspect_messages(candidate_messages)

        best_messages = messages
        best_inspection = self._inspect_messages(messages)
        truncated = False
        lo, hi = 0, len(user_content)
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate_messages, candidate_inspection = _candidate_with_chars(mid)
            if candidate_inspection.get("input_token_over_budget"):
                hi = mid - 1
                continue
            best_messages = candidate_messages
            best_inspection = candidate_inspection
            truncated = mid < len(user_content)
            lo = mid + 1

        if best_inspection.get("input_token_over_budget"):
            empty_messages, empty_inspection = _candidate_with_chars(0)
            return empty_messages, empty_inspection, True

        return best_messages, best_inspection, truncated

    def decide(
            self,
            current_date: str,
            fundamentals: Dict[str, str],
            price_history: Dict[str, Any],
            tsfm_forecasts: Optional[Dict[str, str]] = None,
            current_weights: Optional[Dict[str, float]] = None,
    ) -> PortfolioDecision:
        """做出权重决策"""
        prepared_request = self.prepare_request(
            current_date, fundamentals, price_history,
            tsfm_forecasts, current_weights
        )
        return self.decide_from_request(prepared_request)
