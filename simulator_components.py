from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MAG7_TICKERS
from .portfolio_models import PortfolioState
from .simulator_models import Trade


class TradingCalendar:
    """负责从价格数据中提取交易日序列。"""

    def get_trading_days(
        self,
        start_date: str,
        end_date: str,
        price_data: Dict[str, pd.DataFrame],
    ) -> List[str]:
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

    def get_previous_trading_day(
        self,
        date: str,
        price_data: Dict[str, pd.DataFrame],
    ) -> Optional[str]:
        first_ticker = list(price_data.keys())[0]
        df = price_data[first_ticker]

        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
        elif 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
        else:
            dates = pd.to_datetime(df.index)

        target_dt = pd.to_datetime(date)
        previous_days = dates[dates < target_dt].sort_values()
        if len(previous_days) == 0:
            return None
        return previous_days.iloc[-1].strftime("%Y-%m-%d")


class PriceProvider:
    """负责按日期读取单只股票价格。"""

    def get_price(
        self,
        ticker: str,
        date: str,
        price_data: Dict[str, pd.DataFrame],
    ) -> float:
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
            raise ValueError(
                f"Missing exact price for {ticker} on trading day {date}"
            )

        if 'adjusted_close' in df.columns:
            return float(row['adjusted_close'].iloc[0])
        if 'close' in df.columns:
            return float(row['close'].iloc[0])
        if 'Close' in df.columns:
            return float(row['Close'].iloc[0])

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return float(row[numeric_cols[-1]].iloc[0])


class RebalanceEngine:
    """负责根据目标权重执行再平衡。"""

    def execute(
        self,
        current_state: PortfolioState,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
    ) -> Tuple[PortfolioState, List[Trade]]:
        trades: List[Trade] = []
        total_value = current_state.total_value

        new_positions = {}
        for ticker in MAG7_TICKERS:
            target_value = total_value * target_weights.get(ticker, 0)
            target_shares = target_value / prices[ticker] if prices[ticker] > 0 else 0

            current_shares = current_state.positions.get(ticker, 0)
            diff_shares = target_shares - current_shares

            if abs(diff_shares) > 0.0001:
                trades.append(
                    Trade(
                        date=current_state.date,
                        ticker=ticker,
                        action="buy" if diff_shares > 0 else "sell",
                        shares=abs(diff_shares),
                        price=prices[ticker],
                        value=abs(diff_shares) * prices[ticker],
                    )
                )

            new_positions[ticker] = target_shares

        invested = sum(new_positions[t] * prices[t] for t in MAG7_TICKERS)
        new_cash = total_value - invested

        return (
            PortfolioState(
                date=current_state.date,
                cash=new_cash,
                positions=new_positions,
                prices=prices,
                weights=target_weights,
            ),
            trades,
        )


class PerformanceCalculator:
    """负责组合绩效指标计算。"""

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    def calculate(
        self,
        daily_returns: List[float],
        initial_capital: float,
        final_value: float,
        num_days: int,
    ) -> Dict[str, float]:
        returns = np.array(daily_returns)
        total_return = (final_value - initial_capital) / initial_capital
        annualized_return = (1 + total_return) ** (252 / max(num_days, 1)) - 1

        if len(returns) > 1 and np.std(returns) > 0:
            excess_returns = returns - self.risk_free_rate / 252
            sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
        else:
            sharpe_ratio = 0.0

        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino_ratio = np.sqrt(252) * np.mean(returns - self.risk_free_rate / 252) / np.std(downside_returns)
        else:
            sortino_ratio = 0.0

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
