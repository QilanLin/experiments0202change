from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Optional

import pandas as pd

from .config import CASH_TICKER, MAG7_TICKERS
from .portfolio_models import PortfolioState


@dataclass
class DailyMarketContext:
    """给 LLM 决策用的日级市场上下文。"""

    fundamentals: Dict[str, str]
    price_history: Dict[str, Any]
    tsfm_forecasts: Optional[Dict[str, str]]
    current_weights: Dict[str, float]
    fundamentals_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MarketContextProvider:
    """负责按日期构造 fundamentals / price history / TSFM prompt block。"""

    def __init__(
        self,
        *,
        data_loader,
        get_price_data: Callable[[], Dict[str, pd.DataFrame]],
        get_tsfm_forecasts: Callable[[], Dict[str, Dict[str, Any]]],
        slice_price_df_upto: Callable[[pd.DataFrame, str], pd.DataFrame],
        format_tsfm_for_llm: Optional[Callable[[Any], str]] = None,
        debug: bool = False,
    ):
        self.data_loader = data_loader
        self.get_price_data = get_price_data
        self.get_tsfm_forecasts = get_tsfm_forecasts
        self.slice_price_df_upto = slice_price_df_upto
        self.format_tsfm_for_llm = format_tsfm_for_llm
        self.debug = debug

    def build(self, date: str, state: PortfolioState) -> DailyMarketContext:
        """构造某个交易日的完整市场上下文。"""
        fundamentals, fundamentals_metadata = self._build_fundamentals(date)
        price_history = self._build_price_history(date)
        tsfm_forecasts = self._build_tsfm_forecasts(date)
        current_weights = self._build_current_weights(state)
        return DailyMarketContext(
            fundamentals=fundamentals,
            price_history=price_history,
            tsfm_forecasts=tsfm_forecasts,
            current_weights=current_weights,
            fundamentals_metadata=fundamentals_metadata,
        )

    def _build_fundamentals(self, date: str) -> tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
        fundamentals: Dict[str, str] = {}
        fundamentals_metadata: Dict[str, Dict[str, Any]] = {}
        for ticker in MAG7_TICKERS:
            try:
                snapshot = self.data_loader.get_simple_fundamentals_asof(
                    ticker, date, lag_days=45
                )
                rendered = self.data_loader.format_simple_fundamentals_for_llm(snapshot)
                if self._is_empty_fundamental_snapshot(snapshot, rendered):
                    fundamentals[ticker] = "No fundamental data available"
                    fundamentals_metadata[ticker] = {
                        "status": "no_data",
                        "error": None,
                    }
                else:
                    fundamentals[ticker] = rendered
                    fundamentals_metadata[ticker] = {
                        "status": "ok",
                        "error": None,
                    }
            except Exception as e:
                if self.debug:
                    print(f"  Warning: Failed to get as-of fundamentals for {ticker} on {date}: {e}")
                fundamentals[ticker] = (
                    "Fundamental data unavailable due to fetch error: "
                    f"{type(e).__name__}: {e}"
                )
                fundamentals_metadata[ticker] = {
                    "status": "error",
                    "error": str(e),
                }
        return fundamentals, fundamentals_metadata

    def _is_empty_fundamental_snapshot(self, snapshot: Any, rendered: str) -> bool:
        if not snapshot:
            return True
        return str(rendered).strip() == "No fundamental data available"

    def _build_price_history(self, date: str) -> Dict[str, Any]:
        price_history: Dict[str, Any] = {}
        price_data = self.get_price_data()

        for ticker in MAG7_TICKERS:
            if ticker not in price_data:
                continue

            df = price_data[ticker]
            df_upto = self.slice_price_df_upto(df, date)

            close_col = 'close' if 'close' in df_upto.columns else 'Close'
            price_history[ticker] = df_upto[close_col].tail(30).tolist()

            if self.debug:
                date_col = 'date' if 'date' in df_upto.columns else 'timestamp'
                max_date = pd.to_datetime(df_upto[date_col]).max()
                current_date_dt = pd.to_datetime(date)
                if max_date > current_date_dt:
                    raise ValueError(
                        f"Data leak detected: {ticker} on {date}: "
                        f"max_date={max_date} > current_date={current_date_dt}"
                    )
        return price_history

    def _build_tsfm_forecasts(self, date: str) -> Optional[Dict[str, str]]:
        if self.format_tsfm_for_llm is None:
            return None

        all_forecasts = self.get_tsfm_forecasts()
        if date not in all_forecasts:
            return None

        tsfm_forecasts: Dict[str, str] = {}
        for ticker, forecast in all_forecasts[date].items():
            if ticker in MAG7_TICKERS:
                tsfm_forecasts[ticker] = self.format_tsfm_for_llm(forecast)

        return tsfm_forecasts or None

    def _build_current_weights(self, state: PortfolioState) -> Dict[str, float]:
        current_weights = state.weights.copy()
        if CASH_TICKER not in current_weights:
            current_weights[CASH_TICKER] = 0.0
        return current_weights
