"""
Portfolio Simulator - 组合模拟器

支持：
- Fractional shares（权重交易）
- 每日/每周再平衡
- 完整的交易记录
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import pandas as pd
import numpy as np

from .config import MAG7_TICKERS, CASH_TICKER, ASSET_TICKERS, EXPERIMENT_CONFIG
from .portfolio_agent import PortfolioDecision, PortfolioState


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
    
    # 收益指标
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    
    # 详细记录
    daily_snapshots: List[DailySnapshot] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    decisions: List[PortfolioDecision] = field(default_factory=list)
    
    # 中间输出
    llm_outputs: List[Dict[str, Any]] = field(default_factory=list)
    tsfm_outputs: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
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
            "tsfm_outputs": self.tsfm_outputs,
        }
    
    def save(self, filepath: str):
        """保存结果到JSON"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class PortfolioSimulator:
    """组合模拟器"""
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        rebalance_frequency: str = "daily",  # "daily" or "weekly"
        cash_interest: bool = False,  # 是否对现金计息
    ):
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.risk_free_rate = 0.05  # 年化无风险利率
        self.cash_interest = cash_interest  # 现金计息开关
        
    def _get_trading_days(
        self, 
        start_date: str, 
        end_date: str,
        price_data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """获取交易日列表"""
        # 使用第一个股票的日期作为交易日
        first_ticker = list(price_data.keys())[0]
        df = price_data[first_ticker]
        
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
        elif 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
        else:
            dates = pd.to_datetime(df.index)
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        mask = (dates >= start_dt) & (dates <= end_dt)
        trading_days = dates[mask].sort_values()
        
        return [d.strftime("%Y-%m-%d") for d in trading_days]
    
    def _get_price(
        self, 
        ticker: str, 
        date: str, 
        price_data: Dict[str, pd.DataFrame]
    ) -> float:
        """获取指定日期的收盘价（优先使用复权价格）"""
        df = price_data[ticker]
        
        if 'date' in df.columns:
            date_col = 'date'
        elif 'timestamp' in df.columns:
            date_col = 'timestamp'
        else:
            df = df.reset_index()
            date_col = df.columns[0]
        
        df[date_col] = pd.to_datetime(df[date_col])
        target_date = pd.to_datetime(date)
        
        row = df[df[date_col] == target_date]
        if len(row) == 0:
            # 找最近的日期
            df_sorted = df.sort_values(date_col)
            row = df_sorted[df_sorted[date_col] <= target_date].iloc[-1:]
        
        # 优先顺序：adjusted_close > close（复权后） > Close
        if 'adjusted_close' in df.columns:
            return float(row['adjusted_close'].iloc[0])
        elif 'close' in df.columns:
            return float(row['close'].iloc[0])
        elif 'Close' in df.columns:
            return float(row['Close'].iloc[0])
        else:
            # 使用最后一个数值列
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            return float(row[numeric_cols[-1]].iloc[0])
    
    def _execute_rebalance(
        self,
        current_state: PortfolioState,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
    ) -> Tuple[PortfolioState, List[Trade]]:
        """执行再平衡（CASH 权重决定现金保留量）"""
        trades = []
        total_value = current_state.total_value
        
        # 对 MAG7 计算目标持仓（CASH 不需要买卖）
        new_positions = {}
        for ticker in MAG7_TICKERS:
            target_value = total_value * target_weights.get(ticker, 0)
            target_shares = target_value / prices[ticker] if prices[ticker] > 0 else 0
            
            current_shares = current_state.positions.get(ticker, 0)
            diff_shares = target_shares - current_shares
            
            if abs(diff_shares) > 0.0001:  # 忽略微小变化
                trade = Trade(
                    date=current_state.date,
                    ticker=ticker,
                    action="buy" if diff_shares > 0 else "sell",
                    shares=abs(diff_shares),
                    price=prices[ticker],
                    value=abs(diff_shares) * prices[ticker],
                )
                trades.append(trade)
            
            new_positions[ticker] = target_shares
        
        # 计算新的现金：total_value - 已投资部分
        # 当 CASH 权重 > 0 时会自然留下现金
        invested = sum(new_positions[t] * prices[t] for t in MAG7_TICKERS)
        new_cash = total_value - invested
        
        new_state = PortfolioState(
            date=current_state.date,
            cash=new_cash,
            positions=new_positions,
            prices=prices,
            weights=target_weights,
        )
        
        return new_state, trades
    
    def _calculate_metrics(
        self, 
        daily_returns: List[float],
        initial_capital: float,
        final_value: float,
        num_days: int,
    ) -> Dict[str, float]:
        """计算性能指标"""
        returns = np.array(daily_returns)
        
        # 总收益
        total_return = (final_value - initial_capital) / initial_capital
        
        # 年化收益
        annualized_return = (1 + total_return) ** (252 / max(num_days, 1)) - 1
        
        # Sharpe Ratio
        if len(returns) > 1 and np.std(returns) > 0:
            excess_returns = returns - self.risk_free_rate / 252
            sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
        else:
            sharpe_ratio = 0.0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino_ratio = np.sqrt(252) * np.mean(returns - self.risk_free_rate / 252) / np.std(downside_returns)
        else:
            sortino_ratio = 0.0
        
        # Max Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
        }
    
    def run(
        self,
        experiment_type: str,
        price_data: Dict[str, pd.DataFrame],
        decision_func,  # Callable that returns PortfolioDecision
        start_date: str,
        end_date: str,
    ) -> SimulationResult:
        """运行模拟"""
        trading_days = self._get_trading_days(start_date, end_date, price_data)
        
        if len(trading_days) == 0:
            raise ValueError(f"No trading days found between {start_date} and {end_date}")
        
        # 初始化：MAG7 等权 + CASH=0
        initial_prices = {t: self._get_price(t, trading_days[0], price_data) for t in MAG7_TICKERS}
        initial_weights = {t: 1/7 for t in MAG7_TICKERS}
        initial_weights[CASH_TICKER] = 0.0
        
        # 初始持仓（等权重）
        initial_positions = {}
        for ticker in MAG7_TICKERS:
            target_value = self.initial_capital * initial_weights[ticker]
            initial_positions[ticker] = target_value / initial_prices[ticker]
        
        current_state = PortfolioState(
            date=trading_days[0],
            cash=0,  # 初始现金为0（全部投资）
            positions=initial_positions,
            prices=initial_prices,
            weights=initial_weights,
        )
        
        # 记录
        daily_snapshots = []
        all_trades = []
        all_decisions = []
        llm_outputs = []
        tsfm_outputs = []
        daily_returns = []
        prev_value = self.initial_capital
        
        for i, date in enumerate(trading_days):
            # 更新价格
            prices = {t: self._get_price(t, date, price_data) for t in MAG7_TICKERS}
            current_state.prices = prices
            current_state.date = date
            
            # 现金计息（如果启用）
            if self.cash_interest and current_state.cash > 0:
                # 每日利率 = (1 + 年化利率)^(1/252) - 1
                daily_rate = (1 + self.risk_free_rate) ** (1/252) - 1
                current_state.cash *= (1 + daily_rate)
            
            # 计算当日收益
            current_value = current_state.total_value
            daily_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0
            daily_returns.append(daily_return)
            prev_value = current_value
            
            # 是否需要再平衡
            should_rebalance = (
                self.rebalance_frequency == "daily" or
                (self.rebalance_frequency == "weekly" and i % 5 == 0)
            )
            
            if should_rebalance:
                # 获取决策
                decision = decision_func(date, current_state)
                all_decisions.append(decision)
                
                # 保存LLM输出
                llm_outputs.append({
                    "date": date,
                    "raw_output": decision.raw_llm_output,
                    "weights": decision.weights,
                    "reasoning": decision.reasoning,
                })
                
                # 执行再平衡
                if decision.action == "rebalance":
                    current_state, trades = self._execute_rebalance(
                        current_state, decision.weights, prices
                    )
                    all_trades.extend(trades)
                    
                    # 最小自检日志：确认 CASH 生效
                    invested = sum(current_state.positions.get(t, 0) * prices.get(t, 0) for t in MAG7_TICKERS)
                    print(f"[REBALANCE] Date: {date}")
                    print(f"  Invested: ${invested:,.2f}")
                    print(f"  Cash: ${current_state.cash:,.2f}")
                    print(f"  Total: ${current_state.total_value:,.2f}")
                    print(f"  CASH weight: {decision.weights.get(CASH_TICKER, 0):.4f}")
            
            # 记录快照
            cumulative_return = (current_value - self.initial_capital) / self.initial_capital
            snapshot = DailySnapshot(
                date=date,
                portfolio_value=current_value,
                cash=current_state.cash,
                positions=current_state.positions.copy(),
                prices=prices.copy(),
                weights=current_state.weights.copy(),
                daily_return=daily_return,
                cumulative_return=cumulative_return,
            )
            daily_snapshots.append(snapshot)
        
        # 计算最终指标
        final_value = current_state.total_value
        metrics = self._calculate_metrics(
            daily_returns, self.initial_capital, final_value, len(trading_days)
        )
        
        return SimulationResult(
            experiment_type=experiment_type,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=metrics["total_return"],
            annualized_return=metrics["annualized_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            max_drawdown=metrics["max_drawdown"],
            daily_snapshots=daily_snapshots,
            trades=all_trades,
            decisions=all_decisions,
            llm_outputs=llm_outputs,
            tsfm_outputs=tsfm_outputs,
        )
