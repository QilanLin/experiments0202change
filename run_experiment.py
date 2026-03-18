"""
Run Experiment - 实验运行主脚本

用法:
    python -m experiments.run_experiment --type baseline --debug
    python -m experiments.run_experiment --type tsfm_format_3 --days 30
    python -m experiments.run_experiment --type baseline --debug --mock
    python -m experiments.run_experiment --type tsfm_format_3 --model timesfm
    python -m experiments.run_experiment --type tsfm_format_3 --model moirai2
    python -m experiments.run_experiment --type tsfm_format_3 --model toto
    python -m experiments.run_experiment --type tsfm_7a --model chronos
    python -m experiments.run_experiment --type tsfm_7b --model chronos
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
    ExperimentType, 
    MAG7_TICKERS,
    CASH_TICKER,
    ASSET_TICKERS,
    get_experiment_dir,
)
from .data_loader import AlphaVantageLoader
from .tsfm_forecaster import TSFMForecaster, TSFMForecast, get_forecaster
from .portfolio_agent import PortfolioWeightAgent, PortfolioDecision, PortfolioState
from .simulator import PortfolioSimulator, SimulationResult
from .lmstudio_openai_chat import LMStudioOpenAIChat
from tradingagents_tsfm_modified_v5.tradingagents.llms.local_qwen import LocalQwenChat


class MockLLM:
    """Mock LLM for debugging without GPU"""
    
    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
    
    def invoke(self, messages):
        """返回mock的权重决策"""
        # 使用已设置的随机种子（在文件开头已设置）
        # 生成随机权重（包含 CASH）
        weights = {t: random.random() for t in ASSET_TICKERS}
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        response = f'''```json
{{
  "action": "rebalance",
  "weights": {{{", ".join(f'"{k}": {v:.4f}' for k, v in weights.items())}}},
  "confidence": 0.7,
  "reasoning": "Mock decision for debugging"
}}
```'''
        
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        return MockResponse(response)


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
        
        # 选择LLM
        llm_model = (
            EXPERIMENT_CONFIG["debug_llm"] if debug 
            else EXPERIMENT_CONFIG["production_llm"]
        )
        
        llm_provider = EXPERIMENT_CONFIG.get("llm_provider", "qwen")

        # 初始化LLM
        if use_mock_llm:
            print("Using MockLLM for debugging...")
            self.llm = MockLLM(model_name="mock")
        elif llm_provider == "qwen":
            self.llm = LocalQwenChat(
                model_name=llm_model,
                temperature=0.0,
            )
        else:
            self.llm = LMStudioOpenAIChat(
                model_name=llm_model,
                base_url=EXPERIMENT_CONFIG["lmstudio_base_url"],
                api_key=EXPERIMENT_CONFIG.get("lmstudio_api_key"),
                temperature=0.0,
            )
        
        # 初始化组件
        self.data_loader = AlphaVantageLoader()
        # 使用工厂函数创建 forecaster，支持 Chronos、TimeFM 和 Moirai2
        self.tsfm_forecaster = get_forecaster(model_name=model_name, use_mock=use_mock_llm) if tsfm_format else None
        self.portfolio_agent = PortfolioWeightAgent(
            self.llm, experiment_type, tsfm_format
        )
        self.simulator = PortfolioSimulator(
            initial_capital=EXPERIMENT_CONFIG["initial_capital"],
            rebalance_frequency="daily",
            cash_interest=False,  # 默认不启用现金计息（可配置）
        )
        
        # 数据缓存
        self._price_data: Dict[str, pd.DataFrame] = {}
        self._fundamentals: Dict[str, Dict] = {}
        self._tsfm_forecasts: Dict[str, Dict[str, TSFMForecast]] = {}
        self._price_history_cache: Dict[str, pd.DataFrame] = {}
        self._historical_1d_forecast_cache: Dict[str, Dict[str, Optional[float]]] = {}
        
        # 结果目录
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = get_experiment_dir(experiment_type, self.run_id)
        os.makedirs(self.results_dir, exist_ok=True)
        if self.tsfm_forecaster is not None:
            self.tsfm_forecaster.input_dir = os.path.join(self.results_dir, "tsfm_inputs")
    
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
        )
        
        self._price_data = data["prices"]
        self._fundamentals = data["fundamentals"]
        
        print(f"Loaded price data for {len(self._price_data)} tickers")
        print(f"Loaded fundamentals for {len(self._fundamentals)} tickers")
        
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

    def _get_cached_historical_1d_prediction(self, ticker: str, origin_dt: pd.Timestamp) -> Optional[float]:
        """
        返回某个历史日期 origin_dt 上，基于当时可见信息生成的 1D 预测收益率。
        结果会缓存，避免 format_7a / format_7b 在多个决策日重复回放同一历史日期。
        """
        cache = self._historical_1d_forecast_cache.setdefault(ticker, {})
        origin_date = pd.to_datetime(origin_dt).strftime("%Y-%m-%d")
        if origin_date in cache:
            return cache[origin_date]

        hist = self._get_price_history_df(ticker)
        df_upto = hist[hist["date"] <= pd.to_datetime(origin_dt)].copy()
        if len(df_upto) < 30:
            cache[origin_date] = None
            return None

        prices = df_upto["close"].astype(float)
        prices.index = pd.to_datetime(df_upto["date"])
        forecast = self.tsfm_forecaster.forecast_all_formats(
            prices,
            ticker,
            origin_date,
            save_input=False,
        )
        if forecast.status == "error" or forecast.ratio_1d is None:
            cache[origin_date] = None
            return None

        cache[origin_date] = float(forecast.ratio_1d)
        return cache[origin_date]

    def _compute_historical_reliability(self, ticker: str, forecast_date: str) -> Dict[str, Dict[str, Any]]:
        """
        为 format_7a / format_7b 计算“过去7个已兑现 1D 预测”的可靠性摘要。

        口径：
        - 在当前决策日 forecast_date 之前，取最近 7 个已经兑现的一日预测起点；
        - 对每个历史起点 s，仅使用 s 当天及之前的数据重新生成 1D forecast；
        - 计算预测收益率与真实次日收益率之间的 MSE；
        - 同时给一个 0~1 的归一化 reliability score，便于 LLM 理解。
        """
        hist = self._get_price_history_df(ticker)
        current_dt = pd.to_datetime(forecast_date)
        current_rows = hist.index[hist["date"] == current_dt]
        window_size = int(EXPERIMENT_CONFIG.get("tsfm_reliability_window_size", 7))

        summary = {
            "window_size": window_size,
            "n": 0,
            "mse": 0.0,
            "normalized_mse": 0.0,
            "normalized_reliability_score": 0.0,
            "samples": [],
        }

        if len(current_rows) == 0:
            return {"past_7_resolved_1d": summary}

        current_idx = int(current_rows[0])
        if current_idx == 0:
            return {"past_7_resolved_1d": summary}

        start_idx = max(0, current_idx - window_size)
        origin_indices = list(range(start_idx, current_idx))

        samples: List[Dict[str, Any]] = []
        squared_errors: List[float] = []
        realized_sq_returns: List[float] = []

        for origin_idx in origin_indices:
            origin_row = hist.iloc[origin_idx]
            target_row = hist.iloc[origin_idx + 1]
            origin_dt = pd.to_datetime(origin_row["date"])
            pred_ratio = self._get_cached_historical_1d_prediction(ticker, origin_dt)
            if pred_ratio is None:
                continue

            p_t = float(origin_row["close"])
            p_true = float(target_row["close"])
            true_ratio = (p_true - p_t) / p_t
            sq_err = (pred_ratio - true_ratio) ** 2

            samples.append(
                {
                    "forecast_origin_date": origin_dt.strftime("%Y-%m-%d"),
                    "resolved_target_date": pd.to_datetime(target_row["date"]).strftime("%Y-%m-%d"),
                    "predicted_return_1d": float(pred_ratio),
                    "realized_return_1d": float(true_ratio),
                    "squared_error": float(sq_err),
                }
            )
            squared_errors.append(float(sq_err))
            realized_sq_returns.append(float(true_ratio ** 2))

        if not samples:
            return {"past_7_resolved_1d": summary}

        mse = float(np.mean(squared_errors))
        reference_scale = float(np.mean(realized_sq_returns)) if realized_sq_returns else 0.0
        normalized_mse = float(mse / max(reference_scale, 1e-12))
        normalized_score = float(1.0 / (1.0 + normalized_mse))

        summary.update(
            {
                "n": len(samples),
                "mse": mse,
                "normalized_mse": normalized_mse,
                "normalized_reliability_score": normalized_score,
                "samples": samples,
            }
        )
        return {"past_7_resolved_1d": summary}
    
    def generate_tsfm_forecasts(self, forecast_date: str):
        """生成TSFM预测"""
        if self.tsfm_forecaster is None:
            return
        
        print(f"Generating TSFM forecasts for {forecast_date}...")
        
        forecasts = {}
        for ticker in MAG7_TICKERS:
            if ticker not in self._price_data:
                continue
            
            df = self._price_data[ticker]
            
            # 切片到 forecast_date，避免未来价格泄露
            df_upto = self._slice_price_df_upto(df, forecast_date)
            
            if len(df_upto) < 30:
                print(f"  Warning: {ticker} has insufficient history (< 30 days) up to {forecast_date}, skipping TSFM forecast")
                continue
            
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
            if self.tsfm_format in (7, 8):
                forecast.historical_reliability = self._compute_historical_reliability(
                    ticker=ticker,
                    forecast_date=forecast_date,
                )
            forecasts[ticker] = forecast
            
            # 保存TSFM输出
            forecast_path = os.path.join(
                self.results_dir, "tsfm_outputs", f"{ticker}_{forecast_date}.json"
            )
            os.makedirs(os.path.dirname(forecast_path), exist_ok=True)
            with open(forecast_path, 'w') as f:
                json.dump(forecast.to_dict(), f, indent=2, default=str)
        
        self._tsfm_forecasts[forecast_date] = forecasts
    
    def _create_decision_func(self):
        """创建决策函数供模拟器使用"""
        def decision_func(date: str, state: PortfolioState) -> PortfolioDecision:
            print(f"[STAGE] Starting LLM decision for {date}", flush=True)
            # 准备基本面数据（使用 as-of 基本面，避免未来信息泄露）
            fundamentals = {}
            for ticker in MAG7_TICKERS:
                try:
                    snapshot = self.data_loader.get_simple_fundamentals_asof(
                        ticker, date, lag_days=45
                    )
                    fundamentals[ticker] = self.data_loader.format_simple_fundamentals_for_llm(snapshot)
                except Exception as e:
                    if self.debug:
                        print(f"  Warning: Failed to get as-of fundamentals for {ticker} on {date}: {e}")
                    fundamentals[ticker] = "No fundamental data available"
            
            # 准备价格历史（只针对 MAG7，CASH 不需要价格数据）
            # 修复：切片到 date，避免未来价格泄露
            price_history = {}
            for ticker in MAG7_TICKERS:
                if ticker in self._price_data:
                    df = self._price_data[ticker]
                    
                    # 切片到当前日期
                    df_upto = self._slice_price_df_upto(df, date)
                    
                    # 统一使用复权后的 close 列
                    close_col = 'close' if 'close' in df_upto.columns else 'Close'
                    price_history[ticker] = df_upto[close_col].tail(30).tolist()
                    
                    # Debug 断言：确保没有未来价格泄露
                    if self.debug:
                        date_col = 'date' if 'date' in df_upto.columns else 'timestamp'
                        max_date = pd.to_datetime(df_upto[date_col]).max()
                        current_date_dt = pd.to_datetime(date)
                        if max_date > current_date_dt:
                            raise ValueError(
                                f"Data leak detected: {ticker} on {date}: "
                                f"max_date={max_date} > current_date={current_date_dt}"
                            )
            
            # 准备TSFM预测（如果有，只针对 MAG7）
            tsfm_forecasts = None
            if self.tsfm_format and date in self._tsfm_forecasts:
                tsfm_forecasts = {}
                for ticker, forecast in self._tsfm_forecasts[date].items():
                    if ticker in MAG7_TICKERS:  # 只包含 MAG7
                        tsfm_forecasts[ticker] = self.tsfm_forecaster.format_for_llm(
                            forecast, self.tsfm_format
                        )
            
            # 确保 current_weights 包含 CASH
            current_weights = state.weights.copy()
            if CASH_TICKER not in current_weights:
                current_weights[CASH_TICKER] = 0.0
            
            # 调用Agent
            decision = self.portfolio_agent.decide(
                current_date=date,
                fundamentals=fundamentals,
                price_history=price_history,
                tsfm_forecasts=tsfm_forecasts,
                current_weights=current_weights,
            )
            
            # 最小自检日志（debug 模式）
            if self.debug:
                weights_sum = sum(decision.weights.values())
                print(f"[DEBUG] Date: {date}")
                print(f"  Parsed weights: {decision.weights}")
                print(f"  Weights sum: {weights_sum:.6f}")
                if abs(weights_sum - 1.0) > 0.01:
                    print(f"  WARNING: Weights sum is not 1.0!")
            
            # 保存LLM输出
            llm_output_path = os.path.join(
                self.results_dir, "llm_outputs", f"decision_{date}.json"
            )
            os.makedirs(os.path.dirname(llm_output_path), exist_ok=True)
            with open(llm_output_path, 'w') as f:
                json.dump(decision.to_dict(), f, indent=2, default=str)
            print(f"[STAGE] Saved LLM decision for {date} -> {llm_output_path}", flush=True)
            
            return decision
        
        return decision_func
    
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
            trading_days = self._get_trading_days(start_date, end_date)
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
        
        # 保存结果
        result_path = os.path.join(self.results_dir, "simulation_result.json")
        result.save(result_path)
        print(f"[STAGE] Saved simulation result -> {result_path}", flush=True)
        
        # 打印摘要
        self._print_summary(result)
        
        return result
    
    def _get_trading_days(self, start_date: str, end_date: str) -> list:
        """获取交易日列表"""
        first_ticker = list(self._price_data.keys())[0]
        df = self._price_data[first_ticker]
        
        date_col = 'date' if 'date' in df.columns else 'timestamp'
        dates = pd.to_datetime(df[date_col])
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        mask = (dates >= start_dt) & (dates <= end_dt)
        trading_days = dates[mask].sort_values()
        
        return [d.strftime("%Y-%m-%d") for d in trading_days]
    
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
        choices=["baseline", "tsfm_1", "tsfm_2", "tsfm_3", "tsfm_4", "tsfm_5", "tsfm_6", "tsfm_7a", "tsfm_7b"],
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
    
    # 映射实验类型
    type_mapping = {
        "baseline": (ExperimentType.BASELINE_LLM_ONLY, None),
        "tsfm_1": (ExperimentType.LLM_TSFM_FORMAT_1, 1),
        "tsfm_2": (ExperimentType.LLM_TSFM_FORMAT_2, 2),
        "tsfm_3": (ExperimentType.LLM_TSFM_FORMAT_3, 3),
        "tsfm_4": (ExperimentType.LLM_TSFM_FORMAT_4, 4),
        "tsfm_5": (ExperimentType.LLM_TSFM_FORMAT_5, 5),
        "tsfm_6": (ExperimentType.LLM_TSFM_FORMAT_6, 6),
        "tsfm_7a": (ExperimentType.LLM_TSFM_FORMAT_7A, 7),
        "tsfm_7b": (ExperimentType.LLM_TSFM_FORMAT_7B, 8),
    }
    
    experiment_type, tsfm_format = type_mapping[args.type]
    
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
