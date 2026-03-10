"""
Portfolio Weight Allocation Agent

LLM决策输出：MAG7股票的权重分配
"""

from __future__ import annotations
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

from .config import MAG7_TICKERS, CASH_TICKER, ASSET_TICKERS, EXPERIMENT_CONFIG


@dataclass
class PortfolioDecision:
    """组合权重决策"""
    decision_date: str
    weights: Dict[str, float]  # ticker -> weight (0-1)
    action: str  # "rebalance" or "hold"
    reasoning: str
    confidence: float
    raw_llm_output: str
    prompt: str = ""  # 保存给LLM的完整提示词
    
    # 元数据
    experiment_type: str = ""
    tsfm_format: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def validate(self) -> Tuple[bool, str]:
        """验证权重是否有效"""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            return False, f"Weights sum to {total:.4f}, not 1.0"
        for ticker, w in self.weights.items():
            if w < 0:
                return False, f"Negative weight for {ticker}: {w}"
        return True, "Valid"


@dataclass
class PortfolioState:
    """组合状态"""
    date: str
    cash: float
    positions: Dict[str, float]  # ticker -> shares (fractional)
    prices: Dict[str, float]  # ticker -> price
    weights: Dict[str, float]  # ticker -> weight
    
    @property
    def total_value(self) -> float:
        position_value = sum(
            self.positions.get(t, 0) * self.prices.get(t, 0) 
            for t in MAG7_TICKERS
        )
        return self.cash + position_value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "cash": self.cash,
            "positions": self.positions,
            "prices": self.prices,
            "weights": self.weights,
            "total_value": self.total_value,
        }


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

    def __init__(self, llm, experiment_type: str, tsfm_format: Optional[int] = None):
        self.llm = llm
        self.experiment_type = experiment_type
        self.tsfm_format = tsfm_format

    def _build_context(
            self,
            current_date: str,
            fundamentals: Dict[str, str],
            price_history: Dict[str, Any],
            tsfm_forecasts: Optional[Dict[str, str]] = None,
            current_weights: Optional[Dict[str, float]] = None,
    ) -> str:
        """构建LLM输入上下文"""
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
                    # 显示完整的历史价格序列，让LLM能看到波动
                    change_pct = (prices[-1] / prices[0] - 1) * 100
                    # 格式化价格列表，保留2位小数
                    prices_str = ", ".join([f"{p:.2f}" for p in prices])
                    context_parts.append(f"{ticker} (30d change: {change_pct:+.1f}%): [{prices_str}]")

        if tsfm_forecasts:
            context_parts.append("\n=== TSFM FORECASTS ===")
            for ticker in MAG7_TICKERS:
                if ticker in tsfm_forecasts:
                    context_parts.append(f"\n{tsfm_forecasts[ticker]}")

        return "\n".join(context_parts)

    def _format_prompt_for_saving(self, messages: List[Dict[str, str]]) -> str:
        """将消息列表格式化为字符串以便保存"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            prompt_parts.append(f"=== {role.upper()} ===\n{content}\n")
        return "\n".join(prompt_parts)

    def _parse_llm_output(self, output: str, current_date: str, prompt: str = "") -> PortfolioDecision:
        """解析LLM输出"""
        raw_output = output

        # 尝试提取JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试直接解析
            json_match = re.search(r'\{[^{}]*"weights"[^{}]*\{[^{}]*\}[^{}]*\}', output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fallback: 等权重
                return self._fallback_decision(current_date, raw_output, "Failed to parse JSON", prompt)

        try:
            data = json.loads(json_str)
            llm_weights_raw = data.get("weights", {})

            # 关键：在归一化前判断 LLM 是否显式提供了 CASH
            cash_provided = CASH_TICKER in llm_weights_raw

            if not cash_provided:
                # 路径1：LLM 没有显式给 CASH
                # 先对 MAG7 权重做 clip 到 [0,1]（缺失补 0）
                mag7_weights = {}
                for ticker in MAG7_TICKERS:
                    raw_val = llm_weights_raw.get(ticker, 0.0)
                    mag7_weights[ticker] = max(0.0, min(1.0, float(raw_val)))

                mag7_sum = sum(mag7_weights.values())

                if mag7_sum == 0:
                    # Fallback: MAG7 等权 + CASH=0
                    weights = {t: 1 / 7 for t in MAG7_TICKERS}
                    weights[CASH_TICKER] = 0.0
                elif mag7_sum <= 1.0:
                    # 直接设置 CASH = 1 - mag7_sum，保持 MAG7 原比例（不归一化）
                    weights = mag7_weights.copy()
                    weights[CASH_TICKER] = 1.0 - mag7_sum
                else:
                    # MAG7 总和 > 1，归一化 MAG7 到 1，CASH=0
                    weights = {t: mag7_weights[t] / mag7_sum for t in MAG7_TICKERS}
                    weights[CASH_TICKER] = 0.0
            else:
                # 路径2：LLM 显式给了 CASH
                # 对 8 个资产权重都 clip 到 [0,1]（缺失补 0）
                weights = {}
                for ticker in ASSET_TICKERS:
                    raw_val = llm_weights_raw.get(ticker, 0.0)
                    weights[ticker] = max(0.0, min(1.0, float(raw_val)))

                total = sum(weights.values())

                if total == 0:
                    # Fallback: MAG7 等权 + CASH=0
                    weights = {t: 1 / 7 for t in MAG7_TICKERS}
                    weights[CASH_TICKER] = 0.0
                else:
                    # 将所有 8 个资产一起归一化到和为 1（包含 CASH）
                    weights = {k: v / total for k, v in weights.items()}

            # 确保所有 8 个资产都有权重（防御性检查）
            for ticker in ASSET_TICKERS:
                if ticker not in weights:
                    weights[ticker] = 0.0

            # 最终检查：处理浮点误差（如果偏离 1 超过 1e-6，重新归一化）
            final_sum = sum(weights.values())
            if abs(final_sum - 1.0) > 1e-6:
                weights = {k: v / final_sum for k, v in weights.items()}

            # Debug 日志（可选，便于验证）
            if hasattr(self, 'debug') and self.debug:
                print(f"[DEBUG] _parse_llm_output: cash_provided={cash_provided}, "
                      f"mag7_sum={sum(weights.get(t, 0) for t in MAG7_TICKERS):.6f}, "
                      f"cash={weights.get(CASH_TICKER, 0):.6f}, final_sum={sum(weights.values()):.6f}")

            return PortfolioDecision(
                decision_date=current_date,
                weights=weights,
                action=data.get("action", "rebalance"),
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", 0.5),
                raw_llm_output=raw_output,
                prompt=prompt,
                experiment_type=self.experiment_type,
                tsfm_format=self.tsfm_format,
            )
        except json.JSONDecodeError as e:
            return self._fallback_decision(current_date, raw_output, f"JSON decode error: {e}", prompt)

    def _fallback_decision(self, current_date: str, raw_output: str, error: str, prompt: str = "") -> PortfolioDecision:
        """Fallback: MAG7 等权重分配 + CASH=0"""
        weights = {t: 1 / 7 for t in MAG7_TICKERS}
        weights[CASH_TICKER] = 0.0
        return PortfolioDecision(
            decision_date=current_date,
            weights=weights,
            action="hold",
            reasoning=f"Fallback due to: {error}",
            confidence=0.0,
            raw_llm_output=raw_output,
            prompt=prompt,
            experiment_type=self.experiment_type,
            tsfm_format=self.tsfm_format,
        )

    def decide(
            self,
            current_date: str,
            fundamentals: Dict[str, str],
            price_history: Dict[str, Any],
            tsfm_forecasts: Optional[Dict[str, str]] = None,
            current_weights: Optional[Dict[str, float]] = None,
    ) -> PortfolioDecision:
        """做出权重决策"""
        context = self._build_context(
            current_date, fundamentals, price_history,
            tsfm_forecasts, current_weights
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        # 将提示词转换为字符串格式以便保存
        prompt_str = self._format_prompt_for_saving(messages)

        result = self.llm.invoke(messages)
        output = result.content if hasattr(result, 'content') else str(result)

        return self._parse_llm_output(output, current_date, prompt_str)