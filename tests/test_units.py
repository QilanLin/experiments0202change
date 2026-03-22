from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import MethodType

import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from module_loader import load_module


data_loader_mod = load_module("data_loader")
portfolio_models_mod = load_module("portfolio_models")
simulator_components_mod = load_module("simulator_components")
simulator_models_mod = load_module("simulator_models")

AlphaVantageLoader = data_loader_mod.AlphaVantageLoader
PortfolioDecision = portfolio_models_mod.PortfolioDecision
TradingCalendar = simulator_components_mod.TradingCalendar
SimulationResult = simulator_models_mod.SimulationResult


class AlphaVantageLoaderTests(unittest.TestCase):
    def _build_loader(self) -> AlphaVantageLoader:
        return AlphaVantageLoader.__new__(AlphaVantageLoader)

    def test_load_all_data_skips_overview_fundamentals_when_disabled(self) -> None:
        loader = self._build_loader()
        fundamentals_calls: list[str] = []

        loader.get_daily_prices = MethodType(
            lambda self, ticker, start_date, end_date, lookback_days: pd.DataFrame(
                {"date": ["2025-01-02"], "close": [100.0], "ticker": [ticker]}
            ),
            loader,
        )
        loader.get_fundamentals = MethodType(
            lambda self, ticker, use_cache=True: fundamentals_calls.append(ticker) or {"ticker": ticker},
            loader,
        )

        result = loader.load_all_data(
            tickers=["AAPL", "CASH", "MSFT"],
            start_date="2025-01-01",
            end_date="2025-01-31",
            include_overview_fundamentals=False,
        )

        self.assertEqual(sorted(result["prices"].keys()), ["AAPL", "MSFT"])
        self.assertEqual(result["fundamentals"], {})
        self.assertEqual(fundamentals_calls, [])

    def test_load_all_data_can_include_overview_fundamentals(self) -> None:
        loader = self._build_loader()
        fundamentals_calls: list[str] = []

        loader.get_daily_prices = MethodType(
            lambda self, ticker, start_date, end_date, lookback_days: pd.DataFrame(
                {"date": ["2025-01-02"], "close": [100.0]}
            ),
            loader,
        )
        loader.get_fundamentals = MethodType(
            lambda self, ticker, use_cache=True: fundamentals_calls.append(ticker) or {"ticker": ticker},
            loader,
        )

        result = loader.load_all_data(
            tickers=["AAPL", "MSFT"],
            start_date="2025-01-01",
            end_date="2025-01-31",
            include_overview_fundamentals=True,
        )

        self.assertEqual(sorted(result["fundamentals"].keys()), ["AAPL", "MSFT"])
        self.assertEqual(fundamentals_calls, ["AAPL", "MSFT"])


class TradingCalendarTests(unittest.TestCase):
    def test_get_trading_days_returns_sorted_dates_in_range(self) -> None:
        calendar = TradingCalendar()
        price_data = {
            "AAPL": pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        ["2025-10-06", "2025-10-01", "2025-10-03", "2025-10-08"]
                    ),
                    "close": [103.0, 100.0, 101.0, 104.0],
                }
            )
        }

        trading_days = calendar.get_trading_days(
            start_date="2025-10-02",
            end_date="2025-10-07",
            price_data=price_data,
        )

        self.assertEqual(trading_days, ["2025-10-03", "2025-10-06"])


class SimulationResultTests(unittest.TestCase):
    def test_to_dict_omits_empty_tsfm_outputs(self) -> None:
        result = SimulationResult(
            experiment_type="baseline_llm_only",
            start_date="2025-01-01",
            end_date="2025-01-31",
            initial_capital=1_000_000.0,
            final_value=1_050_000.0,
            total_return=0.05,
            annualized_return=0.05,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown=0.03,
        )

        payload = result.to_dict()

        self.assertNotIn("tsfm_outputs", payload)

    def test_to_dict_includes_tsfm_outputs_when_present(self) -> None:
        result = SimulationResult(
            experiment_type="llm_tsfm_format_2",
            start_date="2025-01-01",
            end_date="2025-01-31",
            initial_capital=1_000_000.0,
            final_value=1_050_000.0,
            total_return=0.05,
            annualized_return=0.05,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown=0.03,
            tsfm_outputs=[{"ticker": "AAPL", "forecast_date": "2025-01-02"}],
        )

        payload = result.to_dict()

        self.assertIn("tsfm_outputs", payload)
        self.assertEqual(payload["tsfm_outputs"][0]["ticker"], "AAPL")


class PortfolioDecisionTests(unittest.TestCase):
    def test_validate_accepts_weights_close_to_one(self) -> None:
        decision = PortfolioDecision(
            decision_date="2025-01-02",
            weights={"AAPL": 0.5, "MSFT": 0.49, "CASH": 0.01},
            action="rebalance",
            reasoning="test",
            confidence=0.8,
            raw_llm_output="{}",
        )

        valid, message = decision.validate()

        self.assertTrue(valid)
        self.assertEqual(message, "Valid")


if __name__ == "__main__":
    unittest.main()
