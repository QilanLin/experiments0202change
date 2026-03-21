from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .config import MAG7_TICKERS, CASH_TICKER


class PromptBuilder:
    """负责构建给 LLM 的 system/user prompt 及其可保存版本。"""

    def build_context(
        self,
        current_date: str,
        fundamentals: Dict[str, str],
        price_history: Dict[str, Any],
        tsfm_forecasts: Optional[Dict[str, str]] = None,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> str:
        """构建 LLM 输入上下文。"""
        # 默认权重：MAG7 等权 + CASH=0
        default_weights = {t: 1 / 7 for t in MAG7_TICKERS}
        default_weights[CASH_TICKER] = 0.0

        context_parts = [
            f"Date: {current_date}",
            f"Current Portfolio Weights: {json.dumps(current_weights or default_weights)}",
            "",
            "=== COMPANY FUNDAMENTALS ===",
        ]

        for ticker in MAG7_TICKERS:
            if ticker in fundamentals:
                context_parts.append(f"\n--- {ticker} ---")
                context_parts.append(fundamentals[ticker][:2000])  # 截断

        context_parts.append("\n=== PRICE HISTORY (Last 30 days) ===")
        for ticker in MAG7_TICKERS:
            if ticker in price_history:
                prices = price_history[ticker]
                if isinstance(prices, list) and len(prices) > 0:
                    # 显示完整的历史价格序列，让 LLM 能看到波动。
                    change_pct = (prices[-1] / prices[0] - 1) * 100
                    prices_str = ", ".join([f"{p:.2f}" for p in prices])
                    context_parts.append(f"{ticker} (30d change: {change_pct:+.1f}%): [{prices_str}]")

        if tsfm_forecasts:
            context_parts.append("\n=== TSFM FORECASTS ===")
            for ticker in MAG7_TICKERS:
                if ticker in tsfm_forecasts:
                    context_parts.append(f"\n{tsfm_forecasts[ticker]}")

        return "\n".join(context_parts)

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
