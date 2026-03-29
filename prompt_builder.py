from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .config import MAG7_TICKERS, CASH_TICKER


class PromptBuilder:
    """负责构建给 LLM 的 system/user prompt 及其可保存版本。"""

    FUNDAMENTALS_HEADER = "=== COMPANY FUNDAMENTALS ==="
    PRICE_HEADER = "=== PRICE HISTORY (Last 30 days) ==="
    TSFM_HEADER = "=== TSFM FORECASTS ==="

    def _default_weights(self) -> Dict[str, float]:
        weights = {t: 1 / 7 for t in MAG7_TICKERS}
        weights[CASH_TICKER] = 0.0
        return weights

    def build_context_sections(
        self,
        current_date: str,
        fundamentals: Dict[str, str],
        price_history: Dict[str, Any],
        tsfm_forecasts: Optional[Dict[str, str]] = None,
        current_weights: Optional[Dict[str, float]] = None,
        asof_date: Optional[str] = None,
    ) -> Dict[str, str | None]:
        """构建结构化上下文 sections，供渲染和截断复用。"""
        prefix_lines = [f"Date: {current_date}"]
        if asof_date and asof_date != current_date:
            prefix_lines.append(f"Market Data As Of: {asof_date}")
        prefix_lines.append(
            f"Current Portfolio Weights: {json.dumps(current_weights or self._default_weights())}"
        )
        prefix = "\n".join(prefix_lines)

        fundamentals_parts: list[str] = []
        for ticker in MAG7_TICKERS:
            if ticker in fundamentals:
                fundamentals_parts.append(f"--- {ticker} ---")
                fundamentals_parts.append(fundamentals[ticker][:2000])  # 截断
                fundamentals_parts.append("")
        if fundamentals_parts and fundamentals_parts[-1] == "":
            fundamentals_parts.pop()
        fundamentals_body = "\n".join(fundamentals_parts)

        price_lines: list[str] = []
        for ticker in MAG7_TICKERS:
            if ticker in price_history:
                prices = price_history[ticker]
                if isinstance(prices, list) and len(prices) > 0:
                    # 显示完整的历史价格序列，让 LLM 能看到波动。
                    change_pct = (prices[-1] / prices[0] - 1) * 100
                    prices_str = ", ".join([f"{p:.2f}" for p in prices])
                    price_lines.append(f"{ticker} (30d change: {change_pct:+.1f}%): [{prices_str}]")
        price_body = "\n".join(price_lines)

        tsfm_body = None
        if tsfm_forecasts:
            tsfm_blocks = [
                str(tsfm_forecasts[ticker])
                for ticker in MAG7_TICKERS
                if ticker in tsfm_forecasts
            ]
            if tsfm_blocks:
                tsfm_body = "\n\n".join(tsfm_blocks)

        return {
            "prefix": prefix,
            "fundamentals": fundamentals_body,
            "price": price_body,
            "tsfm": tsfm_body,
        }

    def render_context_from_sections(self, sections: Dict[str, str | None]) -> str:
        """把结构化 sections 渲染为最终 user context 字符串。"""
        prefix = str(sections.get("prefix") or "").strip("\n")
        fundamentals_body = str(sections.get("fundamentals") or "").strip("\n")
        price_body = str(sections.get("price") or "").strip("\n")
        tsfm_body = sections.get("tsfm")

        context = f"{prefix}\n\n{self.FUNDAMENTALS_HEADER}"
        if fundamentals_body:
            context += f"\n\n{fundamentals_body}"

        context += f"\n\n{self.PRICE_HEADER}"
        if price_body:
            context += f"\n{price_body}"

        if tsfm_body is not None:
            tsfm_body = str(tsfm_body).strip("\n")
            context += f"\n\n{self.TSFM_HEADER}"
            if tsfm_body:
                context += f"\n\n{tsfm_body}"

        return context.strip("\n")

    def build_context(
        self,
        current_date: str,
        fundamentals: Dict[str, str],
        price_history: Dict[str, Any],
        tsfm_forecasts: Optional[Dict[str, str]] = None,
        current_weights: Optional[Dict[str, float]] = None,
        asof_date: Optional[str] = None,
    ) -> str:
        """构建 LLM 输入上下文。"""
        sections = self.build_context_sections(
            current_date=current_date,
            asof_date=asof_date,
            fundamentals=fundamentals,
            price_history=price_history,
            tsfm_forecasts=tsfm_forecasts,
            current_weights=current_weights,
        )
        return self.render_context_from_sections(sections)

    def build_messages(self, system_prompt: str, user_context: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_context},
        ]

    def format_prompt_for_saving(self, messages: List[Dict[str, str]]) -> str:
        """将消息列表格式化为字符串以便保存。"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            prompt_parts.append(f"=== {role.upper()} ===\n{content}\n")
        return "\n".join(prompt_parts)
