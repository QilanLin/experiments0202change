"""
Run Experiment - 实验运行主脚本

用法:
    python -m experiments0202change.run_experiment --type baseline --debug
    python -m experiments0202change.run_experiment --type tsfm_3 --days 30
    python -m experiments0202change.run_experiment --type baseline --debug --mock
    python -m experiments0202change.run_experiment --type tsfm_3 --model timesfm
    python -m experiments0202change.run_experiment --type tsfm_3 --model moirai2
    python -m experiments0202change.run_experiment --type tsfm_3 --model toto
    python -m experiments0202change.run_experiment --type tsfm_7a --model chronos
    python -m experiments0202change.run_experiment --type tsfm_7b --model chronos

服务器推荐：
    bash run_server_experiment.sh --type tsfm_7a --model timesfm --start-date 2025-08-31 --end-date 2025-09-30
"""

from __future__ import annotations
import argparse
import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# 设置随机种子以确保实验可复现
RANDOM_SEED = 123
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
try:
    import torch
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
except ImportError:
    pass

from .config import (
    EXPERIMENT_CONFIG, 
    MAG7_TICKERS,
    get_experiment_dir,
)
from .format_registry import CLI_EXPERIMENT_CHOICES, FORMAT_SPEC_BY_CLI
from .artifact_store import ArtifactStore
from .data_loader import AlphaVantageLoader
from .daily_decision_pipeline import DailyDecisionPipeline
from .historical_reliability import HistoricalReliabilityCalculator
from .llm_clients import build_llm_client
from .market_context import MarketContextProvider
from .tsfm_forecaster import TSFMForecast, get_forecaster
from .portfolio_agent import PortfolioWeightAgent
from .simulator import PortfolioSimulator
from .simulator_models import SimulationResult


class TSFMForecastGenerationError(RuntimeError):
    """Raised when a required TSFM forecast fails during an experiment run."""


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(
        self,
        experiment_type: str,
        debug: bool = True,
        simulation_days: int = 30,
        tsfm_format: Optional[int] = None,
        use_mock_llm: bool = False,
        model_name: str = 'chronos',
    ):
        self.experiment_type = experiment_type
        self.debug = debug
        self.simulation_days = simulation_days
        self.tsfm_format = tsfm_format
        self.use_mock_llm = use_mock_llm
        self.model_name = model_name

        # 初始化 LLM：统一通过客户端工厂构建，runner 不再关心 provider 分支。
        self.llm = build_llm_client(debug=debug, use_mock_llm=use_mock_llm)
        
        # 初始化组件
        self.data_loader = AlphaVantageLoader()
        # 使用工厂函数创建 forecaster，支持 Chronos、TimeFM 和 Moirai2
        self.tsfm_forecaster = (
            get_forecaster(
                model_name=model_name,
                use_mock=use_mock_llm,
                debug=self.debug,
            )
            if tsfm_format
            else None
        )
        self.portfolio_agent = PortfolioWeightAgent(
            self.llm, experiment_type, tsfm_format
        )
        self.simulator = PortfolioSimulator(
            initial_capital=EXPERIMENT_CONFIG["initial_capital"],
            rebalance_frequency="daily",
            cash_interest=False,  # 默认不启用现金计息（可配置）
        )
        self.trading_calendar = self.simulator.trading_calendar
        
        # 数据缓存
        self._price_data: Dict[str, pd.DataFrame] = {}
        self._tsfm_forecasts: Dict[str, Dict[str, TSFMForecast]] = {}
        self._price_history_cache: Dict[str, pd.DataFrame] = {}
        self._historical_reliability: Optional[HistoricalReliabilityCalculator] = None
        
        # 结果目录
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = get_experiment_dir(experiment_type, self.run_id)
        # 统一的运行期产物保存入口：tsfm_inputs / tsfm_outputs / llm_outputs / simulation_result
        self.artifact_store = ArtifactStore(self.results_dir)
        if self.tsfm_forecaster is not None:
            self.tsfm_forecaster.input_dir = self.artifact_store.tsfm_input_dir()
            self.tsfm_forecaster.artifact_store = self.artifact_store
        if self.tsfm_forecaster is not None and self.tsfm_format in (7, 8):
            # format_7a / 7b 专用：历史可靠性逻辑从 runner 抽到独立 calculator。
            self._historical_reliability = HistoricalReliabilityCalculator(
                tsfm_forecaster=self.tsfm_forecaster,
                get_price_history_df=self._get_price_history_df,
                window_size=int(EXPERIMENT_CONFIG.get("tsfm_reliability_window_size", 7)),
            )
        self.market_context_provider = MarketContextProvider(
            data_loader=self.data_loader,
            get_price_data=lambda: self._price_data,
            get_tsfm_forecasts=lambda: self._tsfm_forecasts,
            slice_price_df_upto=self._slice_price_df_upto,
            format_tsfm_for_llm=(
                None
                if self.tsfm_forecaster is None or self.tsfm_format is None
                else lambda forecast: self.tsfm_forecaster.format_for_llm(
                    forecast, self.tsfm_format
                )
            ),
            debug=self.debug,
        )
    
    def load_data(self, end_date: str = None):
        """加载所有数据"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        # 免费API只有最近100天数据，所以lookback设为90天
        start_dt = end_dt - timedelta(days=365)
        start_date = start_dt.strftime("%Y-%m-%d")
        
        print(f"Loading data from {start_date} to {end_date}...")
        
        data = self.data_loader.load_all_data(
            tickers=MAG7_TICKERS,
            start_date=start_date,
            end_date=end_date,
            lookback_days=365,  # 免费API限制
            include_overview_fundamentals=False,
        )
        
        self._price_data = data["prices"]
        
        print(f"Loaded price data for {len(self._price_data)} tickers")
        print("Overview fundamentals preload: skipped (as-of fundamentals are loaded on demand)")
        
        if len(self._price_data) == 0:
            raise ValueError("Failed to load any price data. Check API key and rate limits.")
    
    def _slice_price_df_upto(self, df: pd.DataFrame, asof_date_str: str) -> pd.DataFrame:
        """
        切片价格数据到指定日期（避免未来价格泄露）
        
        参数:
            df: 包含 'date' 列的价格DataFrame
            asof_date_str: 截止日期（字符串，格式：YYYY-MM-DD）
        
        返回:
            切片后的DataFrame，只包含 date <= asof_date 的数据，按日期排序并重置索引
        """
        # 确保 date 列是 datetime 类型
        if 'date' in df.columns:
            date_col = 'date'
        elif 'timestamp' in df.columns:
            date_col = 'timestamp'
        else:
            raise ValueError(f"No date/timestamp column found in DataFrame. Columns: {df.columns.tolist()}")
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        asof_date = pd.to_datetime(asof_date_str)
        
        # 过滤：只保留 date <= asof_date 的数据
        df_sub = df[df[date_col] <= asof_date].copy()
        df_sub = df_sub.sort_values(date_col).reset_index(drop=True)
        
        return df_sub

    def _get_price_history_df(self, ticker: str) -> pd.DataFrame:
        """返回统一字段、按日期升序的价格历史缓存。"""
        if ticker in self._price_history_cache:
            return self._price_history_cache[ticker]

        df = self._price_data[ticker].copy()
        date_col = 'date' if 'date' in df.columns else 'timestamp'
        close_col = 'close' if 'close' in df.columns else 'Close'
        hist = df[[date_col, close_col]].copy()
        hist = hist.rename(columns={date_col: 'date', close_col: 'close'})
        hist['date'] = pd.to_datetime(hist['date'])
        hist['close'] = hist['close'].astype(float)
        hist = hist.sort_values('date').reset_index(drop=True)
        self._price_history_cache[ticker] = hist
        return hist

    def generate_tsfm_forecasts(self, forecast_date: str):
        """生成TSFM预测"""
        if self.tsfm_forecaster is None:
            return
        
        print(f"Generating TSFM forecasts for {forecast_date}...")
        
        forecasts = {}
        required_tickers: list[str] = []
        for ticker in MAG7_TICKERS:
            if ticker not in self._price_data:
                continue
            
            df = self._price_data[ticker]
            
            # 切片到 forecast_date，避免未来价格泄露
            df_upto = self._slice_price_df_upto(df, forecast_date)
            
            if len(df_upto) < 30:
                print(f"  Warning: {ticker} has insufficient history (< 30 days) up to {forecast_date}, skipping TSFM forecast")
                continue
            required_tickers.append(ticker)
            
            # 统一使用复权后的 close 列
            close_col = 'close' if 'close' in df_upto.columns else 'Close'
            prices = df_upto[close_col].astype(float)
            
            # 设置索引为真实日期，而不是数字索引
            date_col = 'date' if 'date' in df_upto.columns else 'timestamp'
            prices.index = pd.to_datetime(df_upto[date_col])
            
            # 验证：确保最后一个日期 <= forecast_date
            if self.debug and len(prices) > 0:
                max_date = prices.index.max()
                forecast_dt = pd.to_datetime(forecast_date)
                if max_date > forecast_dt:
                    raise ValueError(
                        f"Data leak detected for {ticker} on {forecast_date}: "
                        f"max_date={max_date} > forecast_date={forecast_dt}"
                    )
            
            forecast = self.tsfm_forecaster.forecast_all_formats(
                prices, ticker, forecast_date
            )
            if self._historical_reliability is not None:
                forecast.historical_reliability = self._historical_reliability.compute(
                    ticker=ticker,
                    forecast_date=forecast_date,
                )
            forecasts[ticker] = forecast
            # 保存 TSFM 输出（保持原有目录结构和文件名）。
            self.artifact_store.save_tsfm_output(
                forecast,
                ticker=ticker,
                forecast_date=forecast_date,
            )

        failed_tickers = {
            ticker: (forecast.error or f"status={forecast.status}")
            for ticker, forecast in forecasts.items()
            if getattr(forecast, "status", None) not in {"success", "mock"}
        }
        missing_tickers = sorted(set(required_tickers) - set(forecasts))
        if failed_tickers or missing_tickers:
            details: list[str] = []
            if failed_tickers:
                failed_summary = ", ".join(
                    f"{ticker}: {reason}" for ticker, reason in sorted(failed_tickers.items())
                )
                details.append(f"failed=[{failed_summary}]")
            if missing_tickers:
                details.append(f"missing={missing_tickers}")
            raise TSFMForecastGenerationError(
                f"TSFM forecast generation failed for {forecast_date}; "
                + "; ".join(details)
            )
        
        self._tsfm_forecasts[forecast_date] = forecasts
    
    def _create_decision_func(self):
        """创建决策函数供模拟器使用"""
        pipeline = DailyDecisionPipeline(
            market_context_provider=self.market_context_provider,
            portfolio_agent=self.portfolio_agent,
            artifact_store=self.artifact_store,
            debug=self.debug,
        )
        return pipeline
    
    def run(self, end_date: str = None, start_date: str = None) -> SimulationResult:
        """运行实验"""
        if start_date is not None and end_date is not None:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            if start_dt > end_dt:
                raise ValueError(f"start_date ({start_date}) cannot be after end_date ({end_date})")
        else:
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            if start_date is None:
                start_dt = end_dt - timedelta(days=self.simulation_days)
                start_date = start_dt.strftime("%Y-%m-%d")
            else:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = start_dt + timedelta(days=self.simulation_days)
                end_date = end_dt.strftime("%Y-%m-%d")
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"\n{'='*60}")
        print(f"Running Experiment: {self.experiment_type}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Debug Mode: {self.debug}")
        print(f"TSFM Format: {self.tsfm_format}")
        if self.tsfm_forecaster:
            print(f"TSFM Model: {self.model_name}")
        print(f"{'='*60}\n")
        
        # 加载数据
        print("[STAGE] Loading market data", flush=True)
        self.load_data(end_date)
        print("[STAGE] Finished loading market data", flush=True)
        
        # 生成TSFM预测（如果需要）
        if self.tsfm_format:
            # 为每个交易日生成预测
            trading_days = self.trading_calendar.get_trading_days(
                start_date, end_date, self._price_data
            )
            print(f"[STAGE] Generating TSFM forecasts for {len(trading_days)} trading days", flush=True)
            for date in trading_days:
                self.generate_tsfm_forecasts(date)
            print("[STAGE] Finished generating TSFM forecasts", flush=True)
        
        # 运行模拟
        print("[STAGE] Creating decision function", flush=True)
        decision_func = self._create_decision_func()
        print("[STAGE] Starting simulator.run", flush=True)
        result = self.simulator.run(
            experiment_type=self.experiment_type,
            price_data=self._price_data,
            decision_func=decision_func,
            start_date=start_date,
            end_date=end_date,
        )
        print("[STAGE] Finished simulator.run", flush=True)
        
        # 保存最终回测结果。
        result_path = self.artifact_store.save_simulation_result(result)
        print(f"[STAGE] Saved simulation result -> {result_path}", flush=True)
        
        # 打印摘要
        self._print_summary(result)
        
        return result
    
    def _print_summary(self, result: SimulationResult):
        """打印结果摘要"""
        print(f"\n{'='*60}")
        print("EXPERIMENT RESULTS")
        print(f"{'='*60}")
        print(f"Experiment Type: {result.experiment_type}")
        print(f"Period: {result.start_date} to {result.end_date}")
        print(f"Initial Capital: ${result.initial_capital:,.2f}")
        print(f"Final Value: ${result.final_value:,.2f}")
        print(f"\nPerformance Metrics:")
        print(f"  Total Return: {result.total_return*100:.2f}%")
        print(f"  Annualized Return: {result.annualized_return*100:.2f}%")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"  Sortino Ratio: {result.sortino_ratio:.3f}")
        print(f"  Max Drawdown: {result.max_drawdown*100:.2f}%")
        print(f"\nActivity:")
        print(f"  Total Trades: {len(result.trades)}")
        print(f"  Decisions Made: {len(result.decisions)}")
        print(f"\nResults saved to: {self.results_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Run portfolio weight allocation experiment")
    parser.add_argument(
        "--type", 
        type=str, 
        default="baseline",
        choices=list(CLI_EXPERIMENT_CHOICES),
        help="Experiment type"
    )
    parser.add_argument("--debug", action="store_true", help="Use debug LLM (smaller model)")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM for debugging (no GPU needed)")
    parser.add_argument("--days", type=int, default=30, help="Simulation days")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--model",
        type=str,
        default="chronos",
        choices=["chronos", "timesfm", "moirai2", "toto"],
        help="TSFM model to use for forecasting: 'chronos', 'timesfm', 'moirai2', or 'toto' (default: chronos)"
    )
    
    args = parser.parse_args()
    
    spec = FORMAT_SPEC_BY_CLI[args.type]
    experiment_type = spec.experiment_type
    tsfm_format = spec.format_id
    
    runner = ExperimentRunner(
        experiment_type=experiment_type,
        debug=args.debug,
        simulation_days=args.days,
        tsfm_format=tsfm_format,
        use_mock_llm=args.mock,
        model_name=args.model,
    )
    
    result = runner.run(end_date=args.end_date, start_date=args.start_date)
    return result


if __name__ == "__main__":
    main()
