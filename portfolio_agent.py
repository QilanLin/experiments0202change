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
    FUNDAMENTALS_HEADER = PromptBuilder.FUNDAMENTALS_HEADER
    PRICE_HEADER = PromptBuilder.PRICE_HEADER
    TSFM_HEADER = PromptBuilder.TSFM_HEADER

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
            asof_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构造给 LLM 的结构化输入。"""
        context_sections = self.prompt_builder.build_context_sections(
            current_date=current_date,
            asof_date=asof_date,
            fundamentals=fundamentals,
            price_history=price_history,
            tsfm_forecasts=tsfm_forecasts,
            current_weights=current_weights,
        )
        user_context = self.prompt_builder.render_context_from_sections(context_sections)
        messages = self.prompt_builder.build_messages(self.SYSTEM_PROMPT, user_context)
        inspection = self._inspect_messages(messages)
        truncated = False
        truncation_strategy = None
        if inspection.get("input_token_over_budget"):
            budget = inspection.get("input_token_budget")
            if budget is not None:
                messages, inspection, truncation_strategy = self._truncate_user_message_to_fit_budget(
                    messages=messages,
                    budget=int(budget),
                    user_sections=context_sections,
                )
                truncated = truncation_strategy is not None

        prompt_str = self.prompt_builder.format_prompt_for_saving(messages)
        return {
            "decision_date": current_date,
            "market_context_asof_date": asof_date or current_date,
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
        user_sections: dict[str, str | None] | None = None,
    ) -> tuple[list[dict[str, str]], Dict[str, Any], str | None]:
        by_section_messages, by_section_inspection, by_section_strategy = (
            self._truncate_user_message_by_sections(
                messages=messages,
                budget=budget,
                user_sections=user_sections,
            )
        )
        if by_section_strategy is not None:
            return by_section_messages, by_section_inspection, by_section_strategy
        return self._truncate_user_message_by_tail_binary_search(messages=messages, budget=budget)

    def _truncate_user_message_by_sections(
        self,
        *,
        messages: list[dict[str, str]],
        budget: int,
        user_sections: dict[str, str | None] | None = None,
    ) -> tuple[list[dict[str, str]], Dict[str, Any], str | None]:
        user_idx = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
        if user_idx is None:
            return messages, self._inspect_messages(messages), None

        sections = dict(user_sections) if user_sections is not None else None
        if sections is None:
            user_content = str(messages[user_idx].get("content", ""))
            sections = self._parse_user_context_sections(user_content)
        if sections is None:
            return messages, self._inspect_messages(messages), None

        current_messages = deepcopy(messages)
        current_inspection = self._inspect_messages(current_messages)
        if not current_inspection.get("input_token_over_budget"):
            return current_messages, current_inspection, None

        truncated_any = False
        for section_name in ("fundamentals", "price", "tsfm"):
            body = sections.get(section_name)
            if body is None:
                continue
            updated_sections, updated_messages, updated_inspection, section_truncated = (
                self._truncate_single_section_to_fit_budget(
                    base_messages=current_messages,
                    user_idx=user_idx,
                    base_sections=sections,
                    section_name=section_name,
                    budget=budget,
                )
            )
            sections = updated_sections
            current_messages = updated_messages
            current_inspection = updated_inspection
            truncated_any = truncated_any or section_truncated
            if not current_inspection.get("input_token_over_budget"):
                break

        if truncated_any and not current_inspection.get("input_token_over_budget"):
            return current_messages, current_inspection, "section_budgeted_truncation"
        return messages, self._inspect_messages(messages), None

    def _truncate_single_section_to_fit_budget(
        self,
        *,
        base_messages: list[dict[str, str]],
        user_idx: int,
        base_sections: dict[str, str | None],
        section_name: str,
        budget: int,
    ) -> tuple[dict[str, str | None], list[dict[str, str]], Dict[str, Any], bool]:
        section_text = base_sections.get(section_name)
        if not section_text:
            inspection = self._inspect_messages(base_messages)
            return dict(base_sections), base_messages, inspection, False

        lo, hi = 0, len(section_text)
        best_sections = dict(base_sections)
        best_messages = base_messages
        best_inspection = self._inspect_messages(base_messages)
        truncated = False

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate_sections = dict(base_sections)
            candidate_sections[section_name], candidate_truncated = self._truncate_text_with_marker(
                section_text,
                chars_to_keep=mid,
                marker=f"\n\n[TRUNCATED_{section_name.upper()}_FOR_TOKEN_BUDGET]",
            )
            candidate_content = self.prompt_builder.render_context_from_sections(candidate_sections)
            candidate_messages = deepcopy(base_messages)
            candidate_messages[user_idx]["content"] = candidate_content
            candidate_inspection = self._inspect_messages(candidate_messages)

            if candidate_inspection.get("input_token_over_budget"):
                hi = mid - 1
                continue

            best_sections = candidate_sections
            best_messages = candidate_messages
            best_inspection = candidate_inspection
            truncated = candidate_truncated
            lo = mid + 1

        return best_sections, best_messages, best_inspection, truncated

    def _truncate_text_with_marker(
        self,
        text: str,
        *,
        chars_to_keep: int,
        marker: str,
    ) -> tuple[str, bool]:
        if chars_to_keep >= len(text):
            return text, False
        if chars_to_keep <= 0:
            return marker.strip(), True
        truncated = text[:chars_to_keep].rstrip()
        if not truncated:
            return marker.strip(), True
        return f"{truncated}{marker}", True

    def _parse_user_context_sections(self, user_content: str) -> dict[str, str | None] | None:
        fund_idx = user_content.find(self.prompt_builder.FUNDAMENTALS_HEADER)
        price_idx = user_content.find(self.prompt_builder.PRICE_HEADER)
        if fund_idx < 0 or price_idx < 0 or price_idx < fund_idx:
            return None

        tsfm_idx = user_content.find(self.prompt_builder.TSFM_HEADER)
        if tsfm_idx >= 0 and tsfm_idx < price_idx:
            return None

        fund_start = fund_idx + len(self.prompt_builder.FUNDAMENTALS_HEADER)
        price_start = price_idx + len(self.prompt_builder.PRICE_HEADER)
        prefix = user_content[:fund_idx].rstrip("\n")

        if tsfm_idx >= 0:
            price_end = tsfm_idx
            tsfm_start = tsfm_idx + len(self.prompt_builder.TSFM_HEADER)
            tsfm_body = user_content[tsfm_start:].strip("\n")
        else:
            price_end = len(user_content)
            tsfm_body = None

        fundamentals_body = user_content[fund_start:price_idx].strip("\n")
        price_body = user_content[price_start:price_end].strip("\n")
        return {
            "prefix": prefix,
            "fundamentals": fundamentals_body,
            "price": price_body,
            "tsfm": tsfm_body,
        }

    def _render_user_context_sections(self, sections: dict[str, str | None]) -> str:
        return self.prompt_builder.render_context_from_sections(sections)

    def _truncate_user_message_by_tail_binary_search(
        self,
        *,
        messages: list[dict[str, str]],
        budget: int,
    ) -> tuple[list[dict[str, str]], Dict[str, Any], str | None]:
        user_idx = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
        if user_idx is None:
            inspection = self._inspect_messages(messages)
            return messages, inspection, None

        user_content = str(messages[user_idx].get("content", ""))
        if not user_content:
            inspection = self._inspect_messages(messages)
            return messages, inspection, None

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
        lo, hi = 0, len(user_content)
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate_messages, candidate_inspection = _candidate_with_chars(mid)
            if candidate_inspection.get("input_token_over_budget"):
                hi = mid - 1
                continue
            best_messages = candidate_messages
            best_inspection = candidate_inspection
            lo = mid + 1

        if best_inspection.get("input_token_over_budget"):
            empty_messages, empty_inspection = _candidate_with_chars(0)
            return empty_messages, empty_inspection, "user_tail_char_truncation"

        truncated = best_messages[user_idx].get("content", "") != user_content
        strategy = "user_tail_char_truncation" if truncated else None
        return best_messages, best_inspection, strategy

    def decide(
            self,
            current_date: str,
            fundamentals: Dict[str, str],
            price_history: Dict[str, Any],
            tsfm_forecasts: Optional[Dict[str, str]] = None,
            current_weights: Optional[Dict[str, float]] = None,
            asof_date: Optional[str] = None,
    ) -> PortfolioDecision:
        """做出权重决策"""
        prepared_request = self.prepare_request(
            current_date=current_date,
            fundamentals=fundamentals,
            price_history=price_history,
            tsfm_forecasts=tsfm_forecasts,
            current_weights=current_weights,
            asof_date=asof_date,
        )
        return self.decide_from_request(prepared_request)
