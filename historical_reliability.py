from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd


class HistoricalReliabilityCalculator:
    """为 format_7a / format_7b 计算过去 7 个已兑现 1D forecast 的可靠性摘要。"""

    def __init__(
        self,
        *,
        tsfm_forecaster,
        get_price_history_df: Callable[[str], pd.DataFrame],
        window_size: int = 7,
    ):
        self.tsfm_forecaster = tsfm_forecaster
        self.get_price_history_df = get_price_history_df
        self.window_size = int(window_size)
        # 按 ticker -> origin_date 缓存历史 1D forecast，避免多个决策日重复回放同一天。
        self._historical_1d_forecast_cache: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {}

    def _get_cached_historical_1d_prediction(
        self,
        ticker: str,
        origin_dt: pd.Timestamp,
    ) -> Optional[Dict[str, Any]]:
        """
        返回某个历史日期 origin_dt 上，基于当时可见信息生成的 1D 预测收益率。
        结果会缓存，避免 format_7a / format_7b 在多个决策日重复回放同一历史日期。
        """
        cache = self._historical_1d_forecast_cache.setdefault(ticker, {})
        origin_date = pd.to_datetime(origin_dt).strftime("%Y-%m-%d")
        if origin_date in cache:
            return cache[origin_date]

        hist = self.get_price_history_df(ticker)
        df_upto = hist[hist["date"] <= pd.to_datetime(origin_dt)].copy()
        if len(df_upto) < 30:
            cache[origin_date] = None
            return None

        # 用历史起点当天及之前可见的数据重新生成 1D forecast，避免未来信息泄露。
        prices = df_upto["close"].astype(float)
        prices.index = pd.to_datetime(df_upto["date"])
        historical_input_subdir = os.path.join("historical_reliability", ticker)
        historical_input_relative_path = os.path.join(
            historical_input_subdir,
            f"tsfm_input_{ticker}_{origin_date}.json",
        )
        forecast = self.tsfm_forecaster.forecast_all_formats(
            prices,
            ticker,
            origin_date,
            save_input=True,
            input_subdir=historical_input_subdir,
            log_input_save=False,
        )
        if forecast.status == "error" or forecast.ratio_1d is None:
            cache[origin_date] = None
            return None

        cache[origin_date] = {
            "predicted_return_1d": float(forecast.ratio_1d),
            "tsfm_input_path": historical_input_relative_path,
        }
        return cache[origin_date]

    def compute(self, ticker: str, forecast_date: str) -> Dict[str, Dict[str, Any]]:
        """
        口径：
        - 在当前决策日 forecast_date 之前，取最近 7 个已经兑现的一日预测起点；
        - 对每个历史起点 s，仅使用 s 当天及之前的数据重新生成 1D forecast；
        - 计算预测收益率与真实次日收益率之间的 MSE；
        - 同时给一个 0~1 的归一化 reliability score，便于 LLM 理解。
        """
        hist = self.get_price_history_df(ticker)
        current_dt = pd.to_datetime(forecast_date)
        current_rows = hist.index[hist["date"] == current_dt]

        summary = {
            "window_size": self.window_size,
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

        # 只回看当前决策日之前最近 window_size 个已经兑现的一日预测起点。
        start_idx = max(0, current_idx - self.window_size)
        origin_indices = list(range(start_idx, current_idx))

        samples = []
        squared_errors = []
        realized_sq_returns = []

        for origin_idx in origin_indices:
            origin_row = hist.iloc[origin_idx]
            target_row = hist.iloc[origin_idx + 1]
            origin_dt = pd.to_datetime(origin_row["date"])
            pred_record = self._get_cached_historical_1d_prediction(ticker, origin_dt)
            if pred_record is None:
                continue
            pred_ratio = float(pred_record["predicted_return_1d"])

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
                    "tsfm_input_path": pred_record.get("tsfm_input_path"),
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
