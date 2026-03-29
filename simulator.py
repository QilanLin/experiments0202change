"""
Portfolio Simulator - 组合模拟器

支持：
- Fractional shares（权重交易）
- 每日/每周再平衡
- 完整的交易记录
"""

from __future__ import annotations
from typing import Dict, List
import pandas as pd

from .config import MAG7_TICKERS, CASH_TICKER
from .portfolio_models import PortfolioState
from .simulator_models import DailySnapshot, SimulationResult


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
        # Domain 组件：不改变 simulator.run() 的接口，只把职责下沉。
        from .simulator_components import (
            PerformanceCalculator,
            PriceProvider,
            RebalanceEngine,
            TradingCalendar,
        )

        self.trading_calendar = TradingCalendar()
        self.price_provider = PriceProvider()
        self.rebalance_engine = RebalanceEngine()
        self.performance_calculator = PerformanceCalculator(
            risk_free_rate=self.risk_free_rate
        )
    
    def run(
        self,
        experiment_type: str,
        price_data: Dict[str, pd.DataFrame],
        decision_func,  # Callable that returns PortfolioDecision
        start_date: str,
        end_date: str,
    ) -> SimulationResult:
        """运行模拟"""
        trading_days = self.trading_calendar.get_trading_days(
            start_date, end_date, price_data
        )
        
        if len(trading_days) == 0:
            raise ValueError(f"No trading days found between {start_date} and {end_date}")

        # 初始化：组合状态固定在首个执行日之前的最近一个交易日。
        # 这样每日决策只会看到前一交易日信息，不会出现“看完当天收盘再按当天收盘交易”。
        initial_context_date = self.trading_calendar.get_previous_trading_day(
            trading_days[0], price_data
        ) or trading_days[0]
        initial_prices = {
            t: self.price_provider.get_price(t, initial_context_date, price_data)
            for t in MAG7_TICKERS
        }
        initial_weights = {t: 1/7 for t in MAG7_TICKERS}
        initial_weights[CASH_TICKER] = 0.0
        
        # 初始持仓（等权重）
        initial_positions = {}
        for ticker in MAG7_TICKERS:
            target_value = self.initial_capital * initial_weights[ticker]
            initial_positions[ticker] = target_value / initial_prices[ticker]
        
        current_state = PortfolioState(
            date=initial_context_date,
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
        daily_returns = []
        prev_value = self.initial_capital
        
        for i, date in enumerate(trading_days):
            execution_prices = {
                t: self.price_provider.get_price(t, date, price_data)
                for t in MAG7_TICKERS
            }

            execution_state = PortfolioState(
                date=date,
                cash=current_state.cash,
                positions=current_state.positions.copy(),
                prices=execution_prices,
                weights=current_state.actual_weights,
            )

            # 现金计息（如果启用）
            if self.cash_interest and execution_state.cash > 0:
                # 每日利率 = (1 + 年化利率)^(1/252) - 1
                daily_rate = (1 + self.risk_free_rate) ** (1/252) - 1
                execution_state.cash *= (1 + daily_rate)

            # 计算当日收益（按当日执行价 mark-to-market，再决定是否调仓）
            current_value = execution_state.total_value
            daily_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0
            daily_returns.append(daily_return)
            prev_value = current_value
            
            # 是否需要再平衡
            should_rebalance = (
                self.rebalance_frequency == "daily" or
                (self.rebalance_frequency == "weekly" and i % 5 == 0)
            )
            
            if should_rebalance and current_state.date != date:
                # 获取决策
                decision = decision_func(date, current_state, asof_date=current_state.date)
                all_decisions.append(decision)
                
                # 保存LLM输出
                llm_outputs.append({
                    "date": date,
                    "market_context_asof_date": current_state.date,
                    "raw_output": decision.raw_llm_output,
                    "weights": decision.weights,
                    "reasoning": decision.reasoning,
                })
                
                # 执行再平衡
                if decision.action == "rebalance":
                    execution_state, trades = self.rebalance_engine.execute(
                        execution_state, decision.weights, execution_prices
                    )
                    all_trades.extend(trades)
                    
                    # 最小自检日志：确认 CASH 生效
                    invested = sum(
                        execution_state.positions.get(t, 0) * execution_prices.get(t, 0)
                        for t in MAG7_TICKERS
                    )
                    print(f"[REBALANCE] Date: {date}")
                    print(f"  Invested: ${invested:,.2f}")
                    print(f"  Cash: ${execution_state.cash:,.2f}")
                    print(f"  Total: ${execution_state.total_value:,.2f}")
                    print(f"  CASH weight: {decision.weights.get(CASH_TICKER, 0):.4f}")

            current_state = PortfolioState(
                date=execution_state.date,
                cash=execution_state.cash,
                positions=execution_state.positions.copy(),
                prices=execution_state.prices.copy(),
                weights=execution_state.actual_weights,
            )
            
            # 记录快照
            cumulative_return = (current_value - self.initial_capital) / self.initial_capital
            snapshot = DailySnapshot(
                date=date,
                portfolio_value=current_value,
                cash=current_state.cash,
                positions=current_state.positions.copy(),
                prices=execution_prices.copy(),
                weights=current_state.actual_weights.copy(),
                daily_return=daily_return,
                cumulative_return=cumulative_return,
            )
            daily_snapshots.append(snapshot)
        
        # 计算最终指标
        final_value = current_state.total_value
        metrics = self.performance_calculator.calculate(
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
        )
