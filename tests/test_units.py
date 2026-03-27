from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from module_loader import load_module


data_loader_mod = load_module("data_loader")
data_repositories_mod = load_module("data_repositories")
artifact_store_mod = load_module("artifact_store")
daily_decision_pipeline_mod = load_module("daily_decision_pipeline")
decision_parser_mod = load_module("decision_parser")
format_registry_mod = load_module("format_registry")
format_renderers_mod = load_module("format_renderers")
historical_reliability_mod = load_module("historical_reliability")
market_context_mod = load_module("market_context")
portfolio_models_mod = load_module("portfolio_models")
simulator_components_mod = load_module("simulator_components")
simulator_models_mod = load_module("simulator_models")
timesfm_forecaster_mod = load_module("timesfm_forecaster")

AlphaVantageLoader = data_loader_mod.AlphaVantageLoader
PriceRepository = data_repositories_mod.PriceRepository
ArtifactStore = artifact_store_mod.ArtifactStore
DailyDecisionPipeline = daily_decision_pipeline_mod.DailyDecisionPipeline
DecisionParser = decision_parser_mod.DecisionParser
TSFM_QUANTILES = format_registry_mod.TSFM_QUANTILES
HORIZON_SPECS = format_registry_mod.HORIZON_SPECS
RendererContext = format_renderers_mod.RendererContext
Format4Renderer = format_renderers_mod.Format4Renderer
Format5Renderer = format_renderers_mod.Format5Renderer
Format6Renderer = format_renderers_mod.Format6Renderer
HistoricalReliabilityCalculator = historical_reliability_mod.HistoricalReliabilityCalculator
DailyMarketContext = market_context_mod.DailyMarketContext
MarketContextProvider = market_context_mod.MarketContextProvider
PortfolioDecision = portfolio_models_mod.PortfolioDecision
PortfolioState = portfolio_models_mod.PortfolioState
TradingCalendar = simulator_components_mod.TradingCalendar
SimulationResult = simulator_models_mod.SimulationResult
TSFMForecaster = load_module("tsfm_forecaster").TSFMForecaster
validate_timesfm_quantiles_strict = timesfm_forecaster_mod._validate_requested_quantiles_strict


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


class PriceRepositoryTests(unittest.TestCase):
    def test_exact_plain_cache_snapshot_beats_broader_local_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = PriceRepository(client=SimpleNamespace(), cache_dir=tmpdir)
            price_dir = Path(tmpdir) / "price"
            exact_path = price_dir / "AAPL_plain_daily_2025-01-31_2026-01-31.csv"
            broad_path = price_dir / "AAPL_adjusted_2025-01-16_2026-01-16.csv"

            exact_df = pd.DataFrame(
                {
                    "date": ["2025-10-15", "2026-01-02", "2026-01-31"],
                    "close": [100.0, 101.0, 102.0],
                }
            )
            broad_df = pd.DataFrame(
                {
                    "date": ["2025-01-16", "2025-10-15", "2026-01-02", "2026-01-31"],
                    "close": [50.0, 150.0, 151.0, 152.0],
                    "adjusted_close": [50.0, 150.0, 151.0, 152.0],
                }
            )
            exact_df.to_csv(exact_path, index=False)
            broad_df.to_csv(broad_path, index=False)

            # 让 exact cache 变成“旧文件”，覆盖本次修复要防止的回归场景。
            stale_time = 1
            os.utime(exact_path, (stale_time, stale_time))

            loaded = repo.get_daily_prices(
                "AAPL",
                start_date="2025-01-31",
                end_date="2026-01-31",
                use_cache=True,
            )

            self.assertEqual(len(loaded), 3)
            self.assertListEqual(loaded["close"].tolist(), [100.0, 101.0, 102.0])


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


class DecisionParserTests(unittest.TestCase):
    def test_parse_without_cash_adds_cash_residual(self) -> None:
        parser = DecisionParser("llm_tsfm_format_2", tsfm_format=2)
        output = """```json
{
  "action": "rebalance",
  "weights": {"AAPL": 0.2, "MSFT": 0.3},
  "confidence": 0.9,
  "reasoning": "test"
}
```"""

        decision = parser.parse(output, current_date="2025-01-02", prompt="prompt")

        self.assertAlmostEqual(decision.weights["AAPL"], 0.2)
        self.assertAlmostEqual(decision.weights["MSFT"], 0.3)
        self.assertAlmostEqual(decision.weights["CASH"], 0.5)
        self.assertAlmostEqual(sum(decision.weights.values()), 1.0)

    def test_parse_with_explicit_cash_clips_and_normalizes_all_assets(self) -> None:
        parser = DecisionParser("llm_tsfm_format_2", tsfm_format=2)
        output = """{
  "action": "rebalance",
  "weights": {"AAPL": 2.0, "MSFT": -1.0, "CASH": 1.0},
  "confidence": 0.5,
  "reasoning": "test"
}"""

        decision = parser.parse(output, current_date="2025-01-02")

        self.assertAlmostEqual(decision.weights["AAPL"], 0.5)
        self.assertAlmostEqual(decision.weights["MSFT"], 0.0)
        self.assertAlmostEqual(decision.weights["CASH"], 0.5)
        self.assertAlmostEqual(sum(decision.weights.values()), 1.0)

    def test_parse_invalid_output_returns_fallback(self) -> None:
        parser = DecisionParser("baseline_llm_only", tsfm_format=None)

        decision = parser.parse("not valid json at all", current_date="2025-01-02")

        self.assertEqual(decision.action, "hold")
        self.assertIn("Failed to parse JSON", decision.reasoning)
        self.assertAlmostEqual(sum(decision.weights.values()), 1.0)
        self.assertAlmostEqual(decision.weights["CASH"], 0.0)


class QuantileDefinitionTests(unittest.TestCase):
    def test_registry_quantiles_match_timesfm_effective_values(self) -> None:
        self.assertEqual(TSFM_QUANTILES, (0.1, 0.2, 0.5, 0.7, 0.9))

    def test_timesfm_strict_quantile_validation_rejects_unsupported_levels(self) -> None:
        supported = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        with self.assertRaises(ValueError) as ctx:
            validate_timesfm_quantiles_strict([0.05, 0.5, 0.95], supported)
        message = str(ctx.exception)
        self.assertIn("unsupported", message)
        self.assertIn("0.05", message)
        self.assertIn("0.95", message)

    def test_timesfm_strict_quantile_validation_accepts_supported_levels(self) -> None:
        supported = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        result = validate_timesfm_quantiles_strict([0.1, 0.5, 0.9], supported)
        self.assertEqual(result, [0.1, 0.5, 0.9])

    def test_format4_uses_updated_lower_and_upper_percentile_labels(self) -> None:
        context = RendererContext(
            horizon_specs=HORIZON_SPECS,
            quantiles=list(TSFM_QUANTILES),
            quantile_keys=lambda: [str(q) for q in TSFM_QUANTILES],
            quantile_explanations=lambda: ("(lower)", "(median)", "(upper)"),
        )
        forecast = SimpleNamespace(
            ticker="AAPL",
            numeric_quantile_30d={
                "0.1": [101.0],
                "0.2": [102.0],
                "0.5": [105.0],
                "0.7": [107.0],
                "0.9": [109.0],
            },
        )

        rendered = Format4Renderer().render(forecast, context)

        self.assertIn("Median (50%)", rendered)
        self.assertIn("10th percentile", rendered)
        self.assertIn("90th percentile", rendered)
        self.assertNotIn("5th percentile", rendered)
        self.assertNotIn("95th percentile", rendered)

    def test_format5_and_format6_use_updated_quantile_keys(self) -> None:
        context = RendererContext(
            horizon_specs=HORIZON_SPECS,
            quantiles=list(TSFM_QUANTILES),
            quantile_keys=lambda: [str(q) for q in TSFM_QUANTILES],
            quantile_explanations=lambda: ("(lower)", "(median)", "(upper)"),
        )
        ratio_quantile_30d = {
            "0.1": [0.01],
            "0.2": [0.02],
            "0.5": [0.05],
            "0.7": [0.07],
            "0.9": [0.09],
        }
        ratio_quantile_multi = {
            "0.1": {spec.key: 0.01 for spec in HORIZON_SPECS},
            "0.2": {spec.key: 0.02 for spec in HORIZON_SPECS},
            "0.5": {spec.key: 0.05 for spec in HORIZON_SPECS},
            "0.7": {spec.key: 0.07 for spec in HORIZON_SPECS},
            "0.9": {spec.key: 0.09 for spec in HORIZON_SPECS},
        }
        forecast = SimpleNamespace(
            ticker="AAPL",
            ratio_quantile_30d=ratio_quantile_30d,
            ratio_quantile_multi=ratio_quantile_multi,
            status="success",
            error=None,
        )

        rendered_5 = Format5Renderer().render(forecast, context)
        rendered_6 = Format6Renderer().render(forecast, context)

        self.assertIn("0.7 quantile", rendered_5)
        self.assertNotIn("0.75 quantile", rendered_5)
        self.assertIn("10th=", rendered_6)
        self.assertIn("90th=", rendered_6)
        self.assertNotIn("95th", rendered_6)


class TSFMDtypeProtocolTests(unittest.TestCase):
    def test_run_forecast_normalizes_backend_float32_output_to_float64(self) -> None:
        forecaster = TSFMForecaster.__new__(TSFMForecaster)

        class FakeBackend:
            def predict_df(self, context_df, prediction_length, quantile_levels):
                return pd.DataFrame(
                    {
                        "id": ["AAPL", "AAPL"],
                        "timestamp": pd.date_range("2025-01-01", periods=2, freq="D"),
                        "predictions": np.array([101.0, 102.0], dtype=np.float32),
                        "0.1": np.array([100.0, 101.0], dtype=np.float32),
                        "0.5": np.array([101.0, 102.0], dtype=np.float32),
                        "0.9": np.array([102.0, 103.0], dtype=np.float32),
                    }
                )

        forecaster.backend = FakeBackend()
        context_df = pd.DataFrame(
            {
                "id": ["AAPL", "AAPL"],
                "timestamp": pd.date_range("2024-12-30", periods=2, freq="D"),
                "target": [99.0, 100.0],
            }
        )

        pred_df = TSFMForecaster._run_forecast(
            forecaster,
            context_df,
            prediction_length=2,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        self.assertEqual(pred_df["predictions"].dtype, np.float64)
        self.assertEqual(pred_df["0.1"].dtype, np.float64)
        self.assertEqual(pred_df["0.5"].dtype, np.float64)
        self.assertEqual(pred_df["0.9"].dtype, np.float64)

    def test_extract_quantile_always_returns_float64_array(self) -> None:
        forecaster = TSFMForecaster.__new__(TSFMForecaster)
        pred_df = pd.DataFrame({"0.5": np.array([1.0, 2.0], dtype=np.float32)})

        q_values = TSFMForecaster._extract_quantile(forecaster, pred_df, 0.5)

        self.assertEqual(q_values.dtype, np.float64)


class HistoricalReliabilityCalculatorTests(unittest.TestCase):
    def _make_history(self, periods: int = 40) -> pd.DataFrame:
        dates = pd.date_range("2025-01-01", periods=periods, freq="D")
        closes = [100.0 + i for i in range(periods)]
        return pd.DataFrame({"date": dates, "close": closes})

    def test_compute_returns_seven_samples_and_uses_cache(self) -> None:
        hist = self._make_history(periods=40)

        class FakeForecaster:
            def __init__(self):
                self.calls = []

            def forecast_all_formats(
                self,
                prices,
                ticker,
                forecast_date,
                save_input=True,
                input_subdir=None,
                log_input_save=False,
            ):
                self.calls.append(
                    {
                        "ticker": ticker,
                        "forecast_date": forecast_date,
                        "max_seen_date": prices.index.max().strftime("%Y-%m-%d"),
                        "input_subdir": input_subdir,
                    }
                )
                return SimpleNamespace(status="ok", ratio_1d=0.01)

        forecaster = FakeForecaster()
        calc = HistoricalReliabilityCalculator(
            tsfm_forecaster=forecaster,
            get_price_history_df=lambda ticker: hist,
            window_size=7,
        )

        result_1 = calc.compute("AAPL", "2025-02-06")
        result_2 = calc.compute("AAPL", "2025-02-06")
        summary = result_1["past_7_resolved_1d"]

        self.assertEqual(summary["n"], 7)
        self.assertEqual(len(summary["samples"]), 7)
        self.assertGreater(summary["mse"], 0.0)
        self.assertGreaterEqual(summary["normalized_reliability_score"], 0.0)
        self.assertLessEqual(summary["normalized_reliability_score"], 1.0)
        self.assertEqual(len(forecaster.calls), 7)
        self.assertEqual(result_1, result_2)
        self.assertTrue(
            all(call["max_seen_date"] <= call["forecast_date"] for call in forecaster.calls)
        )
        self.assertTrue(
            all("historical_reliability/AAPL" in sample["tsfm_input_path"] for sample in summary["samples"])
        )

    def test_compute_returns_empty_summary_when_history_is_insufficient(self) -> None:
        hist = self._make_history(periods=20)
        calc = HistoricalReliabilityCalculator(
            tsfm_forecaster=SimpleNamespace(
                forecast_all_formats=lambda *args, **kwargs: SimpleNamespace(status="ok", ratio_1d=0.01)
            ),
            get_price_history_df=lambda ticker: hist,
            window_size=7,
        )

        summary = calc.compute("AAPL", "2025-01-15")["past_7_resolved_1d"]

        self.assertEqual(summary["n"], 0)
        self.assertEqual(summary["samples"], [])


class MarketContextProviderTests(unittest.TestCase):
    def _build_provider(self, *, debug: bool = False, slice_fn=None, get_tsfm_forecasts=None):
        class FakeLoader:
            def get_simple_fundamentals_asof(self, ticker, asof_date, lag_days=45):
                return {"ticker": ticker, "asof_date": asof_date}

            def format_simple_fundamentals_for_llm(self, snapshot):
                return f"fundamentals-{snapshot['ticker']}-{snapshot['asof_date']}"

        price_data = {
            "AAPL": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
                    "close": [100.0, 101.0, 102.0, 103.0],
                }
            ),
            "MSFT": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
                    "close": [200.0, 201.0, 202.0],
                }
            ),
        }

        return MarketContextProvider(
            data_loader=FakeLoader(),
            get_price_data=lambda: price_data,
            get_tsfm_forecasts=(
                get_tsfm_forecasts
                if get_tsfm_forecasts is not None
                else lambda: {
                    "2025-01-03": {
                        "AAPL": SimpleNamespace(name="aapl"),
                        "MSFT": SimpleNamespace(name="msft"),
                    }
                }
            ),
            slice_price_df_upto=(
                slice_fn
                if slice_fn is not None
                else lambda df, date: df[df["date"] <= pd.to_datetime(date)].copy()
            ),
            format_tsfm_for_llm=lambda forecast: f"rendered-{forecast.name}",
            debug=debug,
        )

    def test_build_trims_price_history_and_adds_cash_weight(self) -> None:
        provider = self._build_provider()
        state = PortfolioState(
            date="2025-01-03",
            cash=100.0,
            positions={"AAPL": 1.0, "MSFT": 1.0},
            prices={"AAPL": 102.0, "MSFT": 202.0},
            weights={"AAPL": 0.4, "MSFT": 0.6},
        )

        context = provider.build("2025-01-03", state)

        self.assertEqual(context.fundamentals["AAPL"], "fundamentals-AAPL-2025-01-03")
        self.assertEqual(context.price_history["AAPL"], [100.0, 101.0, 102.0])
        self.assertEqual(context.tsfm_forecasts["AAPL"], "rendered-aapl")
        self.assertEqual(context.current_weights["CASH"], 0.0)

    def test_build_raises_when_debug_detects_future_prices(self) -> None:
        provider = self._build_provider(
            debug=True,
            slice_fn=lambda df, date: df.copy(),
            get_tsfm_forecasts=lambda: {},
        )
        state = PortfolioState(
            date="2025-01-02",
            cash=0.0,
            positions={},
            prices={},
            weights={},
        )

        with self.assertRaises(ValueError):
            provider.build("2025-01-02", state)


class DailyDecisionPipelineTests(unittest.TestCase):
    def test_pipeline_saves_structured_llm_input_and_decision_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context = DailyMarketContext(
                fundamentals={"AAPL": "fa"},
                price_history={"AAPL": [100.0, 101.0]},
                tsfm_forecasts={"AAPL": "forecast"},
                current_weights={"AAPL": 0.5, "CASH": 0.5},
            )

            class FakeContextProvider:
                def build(self, date, state):
                    return context

            class FakeAgent:
                experiment_type = "llm_tsfm_format_7a"
                tsfm_format = 7

                def prepare_request(self, **kwargs):
                    return {
                        "decision_date": kwargs["current_date"],
                        "messages": [
                            {"role": "system", "content": "system"},
                            {"role": "user", "content": "user"},
                        ],
                        "prompt": "prompt-text",
                    }

                def decide_from_request(self, prepared_request):
                    return PortfolioDecision(
                        decision_date=prepared_request["decision_date"],
                        weights={"AAPL": 0.5, "MSFT": 0.0, "GOOGL": 0.0, "AMZN": 0.0, "META": 0.0, "TSLA": 0.0, "NVDA": 0.0, "CASH": 0.5},
                        action="rebalance",
                        reasoning="ok",
                        confidence=0.8,
                        raw_llm_output="{}",
                        prompt="prompt-text",
                    )

            pipeline = DailyDecisionPipeline(
                market_context_provider=FakeContextProvider(),
                portfolio_agent=FakeAgent(),
                artifact_store=ArtifactStore(tmpdir),
                debug=False,
            )
            state = PortfolioState(
                date="2025-01-03",
                cash=0.0,
                positions={},
                prices={},
                weights={"AAPL": 0.5, "CASH": 0.5},
            )

            decision = pipeline("2025-01-03", state)

            input_path = Path(tmpdir) / "llm_inputs" / "decision_2025-01-03.json"
            output_path = Path(tmpdir) / "llm_outputs" / "decision_2025-01-03.json"
            self.assertTrue(input_path.exists())
            self.assertTrue(output_path.exists())

            saved_input = json.loads(input_path.read_text(encoding="utf-8"))
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_input["decision_date"], "2025-01-03")
            self.assertEqual(saved_input["market_context"]["fundamentals"]["AAPL"], "fa")
            self.assertEqual(saved_input["messages"][0]["role"], "system")
            self.assertEqual(saved_input["prompt"], "prompt-text")
            self.assertEqual(saved_input["tsfm_format"], 7)
            self.assertEqual(saved["decision_date"], "2025-01-03")
            self.assertEqual(saved["prompt"], "prompt-text")
            self.assertEqual(decision.action, "rebalance")


if __name__ == "__main__":
    unittest.main()
