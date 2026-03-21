from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from .config import CASH_TICKER, MAG7_TICKERS


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
