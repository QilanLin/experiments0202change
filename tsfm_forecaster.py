"""
TSFM Forecaster - Chronos-2 多格式预测模块

支持8种输出格式：
1. 数字，接下来30天
2. 比例，接下来30天
3. 比例，1天/1周/2周/3周/4周
4. 数字，分位数 {0.1, 0.2, 0.5, 0.7, 0.9}，30天
5. 比例，分位数，30天
6. 比例，分位数，多时间窗口
7a. 比例，多时间窗口 + 过去7个已兑现1D预测MSE
7b. 比例，多时间窗口 + 过去7个已兑现1D预测MSE + 归一化分数
"""

from __future__ import annotations
import io
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

from .format_registry import HORIZON_SPECS, TSFM_QUANTILES
from .format_renderers import RendererContext, build_format_renderers
from .tsfm_backends import BaseTSFMBackend, build_tsfm_backend


def get_forecaster(model_name: str = 'chronos', **kwargs):
    """
    模型工厂函数：根据 model_name 返回对应的 forecaster 实例
    
    参数:
        model_name: 模型名称，'chronos'、'timesfm'、'moirai2' 或 'toto'（默认 'chronos'）
        **kwargs: 传递给 TSFMForecaster 的其他参数
    
    返回:
        TSFMForecaster 实例
    
    示例:
        forecaster = get_forecaster('chronos', use_mock=True)
        forecaster = get_forecaster('timesfm', device='cuda')
        forecaster = get_forecaster('moirai2', device='cuda')
        forecaster = get_forecaster('toto', device='cuda')
    """
    if model_name == 'chronos':
        # 保持原有行为：使用 Chronos
        return TSFMForecaster(forecaster_type='chronos', **kwargs)
    elif model_name == 'timesfm':
        # 使用 TimeFM
        return TSFMForecaster(forecaster_type='timesfm', **kwargs)
    elif model_name == 'moirai2':
        # 使用 Moirai2
        return TSFMForecaster(forecaster_type='moirai2', **kwargs)
    elif model_name == 'toto':
        # 使用 Toto
        return TSFMForecaster(forecaster_type='toto', **kwargs)
    else:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Supported values: 'chronos', 'timesfm', 'moirai2', 'toto'"
        )


@dataclass
class TSFMForecast:
    """TSFM预测结果数据类"""
    ticker: str
    forecast_date: str
    format_type: str

    # 格式1: 数字，30天
    numeric_30d: Optional[List[float]] = None

    # 格式2: 比例，30天
    ratio_30d: Optional[List[float]] = None

    # 格式3: 比例，多时间窗口
    ratio_1d: Optional[float] = None
    ratio_1w: Optional[float] = None
    ratio_2w: Optional[float] = None
    ratio_3w: Optional[float] = None
    ratio_4w: Optional[float] = None

    # 格式4: 数字，分位数，30天
    numeric_quantile_30d: Optional[Dict[str, List[float]]] = None

    # 格式5: 比例，分位数，30天
    ratio_quantile_30d: Optional[Dict[str, List[float]]] = None

    # 格式6: 比例，分位数，多时间窗口
    ratio_quantile_multi: Optional[Dict[str, Dict[str, float]]] = None

    # 格式7a/7b: 历史可靠性（基于过去7个已兑现的1D forecast）
    historical_reliability: Optional[Dict[str, Any]] = None

    # 元数据
    last_close: Optional[float] = None
    status: str = "pending"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class TSFMForecaster:
    """时序预测器（支持 Chronos、TimeFM 和 Moirai2）"""

    QUANTILES = list(TSFM_QUANTILES)
    HORIZONS = {spec.key: spec.days for spec in HORIZON_SPECS}

    def __init__(
        self, 
        forecaster_type: str = 'chronos',
        model_name: str = "amazon/chronos-2", 
        device: str = None, 
        use_mock: bool = False,
        input_dir: str = "tsfm_inputs",
        artifact_store = None,
    ):
        """
        初始化时序预测器
        
        参数:
            forecaster_type: 预测器类型，'chronos'、'timesfm'、'moirai2' 或 'toto'（默认 'chronos'）
            model_name: 具体模型名称（如 "amazon/chronos-2"），仅用于 Chronos
            device: 设备（'cuda' 或 'cpu'）
            use_mock: 是否使用 mock 模式
        """
        self.forecaster_type = forecaster_type
        self.model_name = model_name  # Chronos 模型名称
        self.device = device
        self.use_mock = use_mock
        self.input_dir = input_dir
        self.artifact_store = artifact_store
        self._renderers = build_format_renderers()
        
        self.backend: BaseTSFMBackend = build_tsfm_backend(
            forecaster_type,
            device=self.device,
            model_name=self.model_name,
        )

    def _load_pipeline(self):
        """延迟加载预测 pipeline（委托给 backend adapter）。"""
        return self.backend.load_pipeline()

    def _prepare_context(
            self,
            prices: pd.Series,
            ticker: str,
            end_date: datetime
    ) -> pd.DataFrame:
        """
        准备Chronos输入数据
        """
        values = prices.values.astype(float)

        # 确保数据按时间排序
        if isinstance(prices.index, pd.DatetimeIndex):
            if not prices.index.is_monotonic_increasing:
                sorted_idx = prices.index.sort_values()
                sorted_prices = prices.reindex(sorted_idx)
                values = sorted_prices.values.astype(float)

        end_dt = pd.to_datetime(end_date)

        # 从end_date往前推，生成len(values)个规律的交易日时间戳
        timestamps = pd.date_range(end=end_dt, periods=len(values), freq="B")

        if len(timestamps) != len(values):
            if len(timestamps) > len(values):
                timestamps = timestamps[-len(values):]
            else:
                timestamps = pd.date_range(end=end_dt, periods=len(values) * 2, freq="B")
                timestamps = timestamps[-len(values):]
                values = values[-len(timestamps):] if len(timestamps) < len(values) else values

        if len(timestamps) > 0:
            max_ts = pd.to_datetime(timestamps.max())
            if max_ts > end_dt + pd.Timedelta(days=1):
                import warnings
                warnings.warn(
                    f"Warning: Last timestamp {max_ts} exceeds end_date {end_dt} by more than 1 day. "
                    f"This may indicate a data alignment issue for {ticker}."
                )

        return pd.DataFrame({
            "id": [ticker] * len(values),
            "timestamp": timestamps,
            "target": values,
        })

    def _run_forecast(
            self,
            context_df: pd.DataFrame,
            prediction_length: int,
            quantile_levels: List[float]
    ) -> pd.DataFrame:
        """
        运行预测（根据 forecaster_type 选择对应的实现）
        
        参数:
            context_df: 上下文数据 DataFrame（包含 id, timestamp, target 列）
            prediction_length: 预测长度
            quantile_levels: 分位数列表
        
        返回:
            预测结果 DataFrame
        """
        pred_df = self.backend.predict_df(
            context_df,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )
        return self._normalize_prediction_frame_dtype(pred_df, quantile_levels)

    def _normalize_prediction_frame_dtype(
            self,
            pred_df: pd.DataFrame,
            quantile_levels: List[float],
    ) -> pd.DataFrame:
        """
        统一 backend 输出后的数值 dtype：在进入 TSFM 派生计算前固定为 float64。
        """
        numeric_cols = {"predictions"}
        for q in quantile_levels:
            numeric_cols.update({str(q), f"{q:.2f}", f"{q:.1f}", f"{q}"})
        for col in numeric_cols:
            if col in pred_df.columns:
                pred_df[col] = pd.to_numeric(pred_df[col], errors="raise").astype(np.float64)
        return pred_df

    def _extract_quantile(self, df: pd.DataFrame, q: float) -> np.ndarray:
        """从预测结果中提取分位数"""
        for key in (q, str(q), f"{q:.2f}", f"{q:.1f}", f"{q}"):
            if key in df.columns:
                return np.asarray(df[key].values, dtype=np.float64)
        if q == 0.5 and "predictions" in df.columns:
            return np.asarray(df["predictions"].values, dtype=np.float64)
        available_cols = list(df.columns)
        raise KeyError(
            f"Missing quantile column for q={q}. "
            f"Available columns: {available_cols}. "
            f"DataFrame shape: {df.shape}"
        )

    def forecast_all_formats(
            self,
            prices: pd.Series,
            ticker: str,
            forecast_date: str,
            save_input: bool = True,
            input_subdir: Optional[str] = None,
            log_input_save: bool = True,
    ) -> TSFMForecast:
        """生成所有8种格式的预测"""
        result = TSFMForecast(
            ticker=ticker,
            forecast_date=forecast_date,
            format_type="all",
        )

        try:
            end_dt = datetime.strptime(forecast_date, "%Y-%m-%d")
            last_close = float(prices.iloc[-1])
            result.last_close = last_close

            context_df = self._prepare_context(prices, ticker, end_dt)

            context_dict = {
                "ticker": ticker,
                "forecast_date": forecast_date,
                "end_date": end_dt.isoformat(),
                "use_mock": self.use_mock,
                "data_points": [
                    {
                        "id": row["id"],
                        "timestamp": row["timestamp"].isoformat() if pd.notna(row["timestamp"]) else None,
                        "target": float(row["target"]) if pd.notna(row["target"]) else None
                    }
                    for _, row in context_df.iterrows()
                ],
                "num_points": len(context_df),
                "last_timestamp": context_df["timestamp"].max().isoformat() if len(context_df) > 0 else None,
                "last_value": float(context_df["target"].iloc[-1]) if len(context_df) > 0 else None
            }
            if save_input:
                if self.artifact_store is not None:
                    input_filename = self.artifact_store.save_tsfm_input(
                        context_dict,
                        ticker=ticker,
                        forecast_date=forecast_date,
                        input_subdir=input_subdir,
                    )
                else:
                    target_input_dir = self.input_dir
                    if input_subdir:
                        target_input_dir = os.path.join(self.input_dir, input_subdir)
                    os.makedirs(target_input_dir, exist_ok=True)
                    input_filename = os.path.join(
                        target_input_dir, f"tsfm_input_{ticker}_{forecast_date}.json"
                    )
                    with open(input_filename, 'w', encoding='utf-8') as f:
                        json.dump(context_dict, f, indent=2, default=str)
                if log_input_save:
                    print(f"[INFO] TSFM输入已保存到: {input_filename}")

            if self.use_mock:
                result = self._generate_mock_forecast(result, last_close)
                result.status = "mock"
                return result

            pred_30d = self._run_forecast(context_df, 30, self.QUANTILES)

            if self.use_mock is False:
                print(f"[DEBUG] TSFM预测结果列名: {list(pred_30d.columns)}")
                print(f"[DEBUG] TSFM预测结果形状: {pred_30d.shape}")
                print(f"[DEBUG] TSFM预测结果前5行:\n{pred_30d.head()}")

            median_30d = self._extract_quantile(pred_30d, 0.5)

            if len(median_30d) == 0:
                raise ValueError(f"预测结果为空，median_30d长度为0")
            if len(median_30d) < 30:
                print(f"[WARNING] 预测结果长度不足30天: {len(median_30d)}")

            result.numeric_30d = median_30d.tolist()
            result.ratio_30d = ((median_30d - last_close) / last_close).tolist()
            # 保持 7dd225a 时代的行为：顶层 ratio_* 直接走 numpy scalar 路径，
            # 避免先转 list 再回填时改变保存到 tsfm_outputs 的数值表示。
            self._assign_ratio_horizons_from_values(result, median_30d, last_close)

            result.numeric_quantile_30d = {}
            for q in self.QUANTILES:
                q_values = self._extract_quantile(pred_30d, q)
                result.numeric_quantile_30d[str(q)] = q_values.tolist()

            result.ratio_quantile_30d = {}
            for q in self.QUANTILES:
                q_values = self._extract_quantile(pred_30d, q)
                result.ratio_quantile_30d[str(q)] = ((q_values - last_close) / last_close).tolist()

            result.ratio_quantile_multi = {}
            for q in self.QUANTILES:
                q_values = self._extract_quantile(pred_30d, q)
                result.ratio_quantile_multi[str(q)] = self._build_ratio_quantile_multi(
                    q_values,
                    last_close,
                )

            result.status = "success"

        except Exception as e:
            result.status = "error"
            result.error = str(e)
            import traceback
            error_trace = traceback.format_exc()
            print(f"[ERROR] TSFM预测失败 for {ticker} on {forecast_date}:")
            print(f"  错误信息: {str(e)}")
            print(f"  错误堆栈:\n{error_trace}")
            if result.last_close is None and len(prices) > 0:
                result.last_close = float(prices.iloc[-1])
            self._apply_fallback(result)

        return result

    def _generate_mock_forecast(self, result: TSFMForecast, last_close: float) -> TSFMForecast:
        """生成mock预测数据用于调试"""
        import random
        # 使用已设置的随机种子（在 run_experiment.py 开头已设置）
        # 如果在此模块独立运行，确保种子已设置
        random.seed(123)

        trend = random.uniform(-0.05, 0.05)
        volatility = random.uniform(0.01, 0.03)

        result.numeric_30d = []
        for i in range(30):
            daily_return = trend / 30 + random.gauss(0, volatility)
            price = last_close * (1 + daily_return * (i + 1))
            result.numeric_30d.append(price)

        result.ratio_30d = [(p - last_close) / last_close for p in result.numeric_30d]
        self._assign_ratio_horizons(result, result.ratio_30d)

        result.numeric_quantile_30d = {}
        for q in self.QUANTILES:
            offset = (q - 0.5) * 2 * volatility * 30
            result.numeric_quantile_30d[str(q)] = [
                p * (1 + offset) for p in result.numeric_30d
            ]

        result.ratio_quantile_30d = {}
        for q in self.QUANTILES:
            result.ratio_quantile_30d[str(q)] = [
                (p - last_close) / last_close
                for p in result.numeric_quantile_30d[str(q)]
            ]

        result.ratio_quantile_multi = {}
        for q in self.QUANTILES:
            quantile_prices = np.array(result.numeric_quantile_30d[str(q)], dtype=float)
            result.ratio_quantile_multi[str(q)] = self._build_ratio_quantile_multi(
                quantile_prices,
                last_close,
            )

        return result

    def _apply_fallback(self, result: TSFMForecast):
        """应用fallback预测（假设价格不变）"""
        if result.last_close is None:
            return

        lc = result.last_close
        result.numeric_30d = [lc] * 30
        result.ratio_30d = [0.0] * 30
        result.ratio_1d = 0.0
        result.ratio_1w = 0.0
        result.ratio_2w = 0.0
        result.ratio_3w = 0.0
        result.ratio_4w = 0.0
        result.numeric_quantile_30d = {str(q): [lc] * 30 for q in self.QUANTILES}
        result.ratio_quantile_30d = {str(q): [0.0] * 30 for q in self.QUANTILES}
        result.ratio_quantile_multi = {
            str(q): {"1d": 0.0, "1w": 0.0, "2w": 0.0, "3w": 0.0, "4w": 0.0}
            for q in self.QUANTILES
        }

    def _get_quantile_explanations(self) -> tuple[str, str, str]:
        quantiles = sorted(float(q) for q in self.QUANTILES)
        lower_q = quantiles[0]
        upper_q = quantiles[-1]
        median_q = min(quantiles, key=lambda q: abs(q - 0.5))

        def _fmt(q: float) -> str:
            return f"{q:g}"

        def _pct(q: float) -> str:
            return f"{q * 100:.0f}"

        expl_lower = (
            f"({_fmt(lower_q)} indicates a {_pct(lower_q)}% probability "
            "of being less than this value)"
        )
        expl_median = (
            f"({_fmt(median_q)} indicates a {_pct(median_q)}% probability "
            "of being less than this value)"
        )
        expl_upper = (
            f"({_fmt(upper_q)} indicates a {_pct(upper_q)}% probability "
            "of being less than this value)"
        )
        return expl_lower, expl_median, expl_upper

    def _quantile_keys(self) -> list[str]:
        return [str(q) for q in self.QUANTILES]

    def _assign_ratio_horizons(self, forecast: TSFMForecast, ratios: List[float]) -> None:
        for spec in HORIZON_SPECS:
            setattr(forecast, spec.ratio_attr, ratios[spec.days - 1])

    def _assign_ratio_horizons_from_values(
        self,
        forecast: TSFMForecast,
        horizon_values: np.ndarray,
        last_close: float,
    ) -> None:
        for spec in HORIZON_SPECS:
            setattr(
                forecast,
                spec.ratio_attr,
                (horizon_values[spec.days - 1] - last_close) / last_close,
            )

    def _build_ratio_quantile_multi(self, q_values: np.ndarray, last_close: float) -> Dict[str, float]:
        return {
            spec.key: (q_values[spec.days - 1] - last_close) / last_close
            for spec in HORIZON_SPECS
        }

    def format_for_llm(self, forecast: TSFMForecast, format_type: int) -> str:
        """将预测结果格式化为LLM可读的字符串"""
        renderer = self._renderers.get(format_type)
        if renderer is None:
            return f"Unknown format type: {format_type}"
        context = RendererContext(
            horizon_specs=HORIZON_SPECS,
            quantiles=self.QUANTILES,
            quantile_keys=self._quantile_keys,
            quantile_explanations=self._get_quantile_explanations,
        )
        return renderer.render(forecast, context)
