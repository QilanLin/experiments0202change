from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from .portfolio_models import PortfolioDecision


@dataclass
class Trade:
    """单笔交易记录"""
    date: str
    ticker: str
    action: str  # "buy" or "sell"
    shares: float
    price: float
    value: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DailySnapshot:
    """每日快照"""
    date: str
    portfolio_value: float
    cash: float
    positions: Dict[str, float]
    prices: Dict[str, float]
    weights: Dict[str, float]
    daily_return: float
    cumulative_return: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationResult:
    """模拟结果"""
    experiment_type: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float

    daily_snapshots: List[DailySnapshot] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    decisions: List[PortfolioDecision] = field(default_factory=list)

    llm_outputs: List[Dict[str, Any]] = field(default_factory=list)
    # TSFM 明细现在由 ArtifactStore 单独落到 results_dir/tsfm_outputs，
    # 不再默认内嵌进 simulation_result.json，避免结果文件里长期出现空字段。
    tsfm_outputs: List[Dict[str, Any]] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "experiment_type": self.experiment_type,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_value": self.final_value,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "num_trades": len(self.trades),
            "num_decisions": len(self.decisions),
            "daily_snapshots": [s.to_dict() for s in self.daily_snapshots],
            "trades": [t.to_dict() for t in self.trades],
            "decisions": [d.to_dict() for d in self.decisions],
            "llm_outputs": self.llm_outputs,
        }
        if self.tsfm_outputs:
            payload["tsfm_outputs"] = self.tsfm_outputs
        return payload

    def save(self, filepath: str):
        """保存结果到JSON"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
