"""
TSFM Forecaster - Chronos-2 多格式预测模块

支持8种输出格式：
1. 数字，接下来30天
2. 比例，接下来30天
3. 比例，1天/1周/2周/3周/4周
4. 数字，分位数 {0.05, 0.25, 0.5, 0.75, 0.95}，30天
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

# Chronos-2 导入
try:
    from chronos import Chronos2Pipeline

    _CHRONOS_AVAILABLE = True
except ImportError:
    Chronos2Pipeline = None
    _CHRONOS_AVAILABLE = False

# TimeFM 导入
try:
    from .timesfm_forecaster import TimesFMForecaster, TimesFMConfig
    _TIMESFM_AVAILABLE = True
except ImportError:
    TimesFMForecaster = None
    TimesFMConfig = None
    _TIMESFM_AVAILABLE = False

# Moirai2 导入
try:
    from .moirai2_forecaster import Moirai2Forecaster, Moirai2Config
    _MOIRAI2_AVAILABLE = True
except ImportError:
    Moirai2Forecaster = None
    Moirai2Config = None
    _MOIRAI2_AVAILABLE = False

# Toto 导入
try:
    from .toto_forecaster import TotoForecasterWrapper, TotoConfig
    _TOTO_AVAILABLE = True
except ImportError:
    TotoForecasterWrapper = None
    TotoConfig = None
    _TOTO_AVAILABLE = False

_CHRONOS_PIPELINE = None


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

    QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
    HORIZONS = {
        "1d": 1,
        "1w": 5,
        "2w": 10,
        "3w": 15,
        "4w": 20,
    }

    def __init__(
        self, 
        forecaster_type: str = 'chronos',
        model_name: str = "amazon/chronos-2", 
        device: str = None, 
        use_mock: bool = False,
        input_dir: str = "tsfm_inputs",
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
        self._pipeline = None
        
        # 验证 forecaster_type
        if forecaster_type not in ('chronos', 'timesfm', 'moirai2', 'toto'):
            raise ValueError(
                f"Invalid forecaster_type: {forecaster_type}. "
                f"Supported values: 'chronos', 'timesfm', 'moirai2', 'toto'"
            )
        
        # TimeFM 可用性检查
        if forecaster_type == 'timesfm' and not _TIMESFM_AVAILABLE:
            raise ImportError(
                f"TimeFM forecaster is not available. "
                f"Please install timesfm package: pip install timesfm"
            )
        
        # Moirai2 可用性检查
        if forecaster_type == 'moirai2' and not _MOIRAI2_AVAILABLE:
            raise ImportError(
                f"Moirai2 forecaster is not available. "
                f"Please install uni2ts package: pip install uni2ts"
            )
        
        # Toto 可用性检查
        if forecaster_type == 'toto' and not _TOTO_AVAILABLE:
            raise ImportError(
                f"Toto forecaster is not available. "
                f"Please install toto package: pip install toto"
            )

    def _load_pipeline(self):
        """延迟加载预测 pipeline（根据 forecaster_type 选择）"""
        if self.forecaster_type == 'chronos':
            return self._load_chronos_pipeline()
        elif self.forecaster_type == 'timesfm':
            return self._load_timesfm_pipeline()
        elif self.forecaster_type == 'moirai2':
            return self._load_moirai2_pipeline()
        elif self.forecaster_type == 'toto':
            return self._load_toto_pipeline()
        else:
            raise ValueError(f"Unknown forecaster_type: {self.forecaster_type}")
    
    def _load_timesfm_pipeline(self):
        """延迟加载TimeFM pipeline"""
        if not _TIMESFM_AVAILABLE:
            raise ImportError("timesfm package not installed")
        
        if self._pipeline is None:
            # 创建 TimeFM forecaster 实例
            cfg = TimesFMConfig(device=self.device)
            self._pipeline = TimesFMForecaster(cfg=cfg, device=self.device)
        
        return self._pipeline
    
    def _load_moirai2_pipeline(self):
        """延迟加载Moirai2 pipeline"""
        if not _MOIRAI2_AVAILABLE:
            raise ImportError("uni2ts package not installed")
        
        if self._pipeline is None:
            # 创建 Moirai2 forecaster 实例
            cfg = Moirai2Config(device=self.device)
            self._pipeline = Moirai2Forecaster(cfg=cfg, device=self.device)
        
        return self._pipeline
    
    def _load_toto_pipeline(self):
        """延迟加载Toto pipeline"""
        if not _TOTO_AVAILABLE:
            raise ImportError("toto package not installed")
        
        if self._pipeline is None:
            # 创建 Toto forecaster 实例
            cfg = TotoConfig(device=self.device)
            self._pipeline = TotoForecasterWrapper(cfg=cfg, device=self.device)
        
        return self._pipeline
    
    def _load_chronos_pipeline(self):
        """延迟加载Chronos pipeline"""
        global _CHRONOS_PIPELINE

        if not _CHRONOS_AVAILABLE:
            raise ImportError("chronos-forecasting not installed")

        if _CHRONOS_PIPELINE is None:
            try:
                import torch
                device_map = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            except ImportError:
                device_map = "cpu"

            _CHRONOS_PIPELINE = Chronos2Pipeline.from_pretrained(
                self.model_name, device_map=device_map
            )

        self._pipeline = _CHRONOS_PIPELINE
        return self._pipeline

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
        if self.forecaster_type == 'chronos':
            return self._run_chronos_forecast(
                context_df, prediction_length, quantile_levels
            )
        elif self.forecaster_type == 'timesfm':
            return self._run_timesfm_forecast(
                context_df, prediction_length, quantile_levels
            )
        elif self.forecaster_type == 'moirai2':
            return self._run_moirai2_forecast(
                context_df, prediction_length, quantile_levels
            )
        elif self.forecaster_type == 'toto':
            return self._run_toto_forecast(
                context_df, prediction_length, quantile_levels
            )
        else:
            raise ValueError(f"Unknown forecaster_type: {self.forecaster_type}")
    
    def _run_timesfm_forecast(
            self,
            context_df: pd.DataFrame,
            prediction_length: int,
            quantile_levels: List[float]
    ) -> pd.DataFrame:
        """
        运行TimeFM预测
        
        TimeFM 只支持 0.1-0.9 的十分位数，需要将 Chronos 的分位数映射到 TimeFM 支持的值。
        映射规则：0.05->0.1, 0.25->0.2, 0.5->0.5, 0.75->0.8, 0.95->0.9
        """
        pipeline = self._load_pipeline()
        
        # TimeFM 支持的分位数：0.1, 0.2, ..., 0.9
        timesfm_supported = {round(i / 10, 1) for i in range(1, 10)}
        
        # 映射规则：将 Chronos 的分位数映射到最接近的 TimeFM 支持值
        quantile_mapping = {
            0.05: 0.1,
            0.25: 0.2,
            0.5: 0.5,
            0.75: 0.8,
            0.95: 0.9,
        }
        
        # 将 quantile_levels 映射到 TimeFM 支持的值
        mapped_quantiles = []
        quantile_to_mapped = {}
        for q in quantile_levels:
            q_float = float(q)
            # 如果已经在支持列表中，直接使用
            if round(q_float, 1) in timesfm_supported:
                mapped_q = round(q_float, 1)
            # 否则使用映射表
            elif q_float in quantile_mapping:
                mapped_q = quantile_mapping[q_float]
            else:
                # 对于其他值，四舍五入到最接近的十分位数
                mapped_q = round(round(q_float, 1) * 10) / 10
                if mapped_q < 0.1:
                    mapped_q = 0.1
                elif mapped_q > 0.9:
                    mapped_q = 0.9
            
            if mapped_q not in mapped_quantiles:
                mapped_quantiles.append(mapped_q)
            quantile_to_mapped[q_float] = mapped_q
        
        # 调用 TimeFM 预测
        pred_df = pipeline.predict_df(
            context_df,
            future_df=None,  # TimeFM 暂不支持 future_df
            prediction_length=prediction_length,
            quantile_levels=mapped_quantiles,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        
        # 将列名从映射后的分位数改回原始请求的分位数
        # 例如：如果请求 0.05，但 TimeFM 返回了 0.1，则将 "0.1" 列重命名为 "0.05"
        # 注意：多个原始分位数可能映射到同一个 TimeFM 分位数（例如 0.05 和 0.1 都映射到 0.1）
        # 这种情况下，我们需要为每个原始分位数创建副本
        
        # 首先，找出哪些映射后的列会被多个原始分位数使用
        mapped_to_orig = {}
        for orig_q, mapped_q in quantile_to_mapped.items():
            mapped_col = f"{mapped_q:.1f}"
            if mapped_col not in mapped_to_orig:
                mapped_to_orig[mapped_col] = []
            mapped_to_orig[mapped_col].append(orig_q)
        
        # 对于每个原始分位数，决定是重命名还是复制
        for orig_q, mapped_q in quantile_to_mapped.items():
            mapped_col = f"{mapped_q:.1f}"
            orig_col = str(orig_q)
            
            # 如果列名相同，不需要处理
            if mapped_col == orig_col:
                continue
            
            if mapped_col in pred_df.columns:
                # 如果这个映射后的列只被一个原始分位数使用，直接重命名
                if len(mapped_to_orig[mapped_col]) == 1:
                    pred_df = pred_df.rename(columns={mapped_col: orig_col})
                else:
                    # 如果被多个原始分位数使用，需要复制列
                    pred_df[orig_col] = pred_df[mapped_col].copy()
        
        return pred_df
    
    def _run_moirai2_forecast(
            self,
            context_df: pd.DataFrame,
            prediction_length: int,
            quantile_levels: List[float]
    ) -> pd.DataFrame:
        """运行Moirai2预测"""
        pipeline = self._load_pipeline()

        return pipeline.predict_df(
            context_df,
            future_df=None,  # Moirai2 暂不支持 future_df
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
    
    def _run_toto_forecast(
            self,
            context_df: pd.DataFrame,
            prediction_length: int,
            quantile_levels: List[float]
    ) -> pd.DataFrame:
        """运行Toto预测"""
        pipeline = self._load_pipeline()

        return pipeline.predict_df(
            context_df,
            future_df=None,  # Toto 暂不支持 future_df
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
    
    def _run_chronos_forecast(
            self,
            context_df: pd.DataFrame,
            prediction_length: int,
            quantile_levels: List[float]
    ) -> pd.DataFrame:
        """运行Chronos预测"""
        pipeline = self._load_pipeline()

        return pipeline.predict_df(
            context_df,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

    def _extract_quantile(self, df: pd.DataFrame, q: float) -> np.ndarray:
        """从预测结果中提取分位数"""
        for key in (q, str(q), f"{q:.2f}", f"{q:.1f}", f"{q}"):
            if key in df.columns:
                return df[key].values
        if q == 0.5 and "predictions" in df.columns:
            return df["predictions"].values
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

            result.ratio_1d = (median_30d[0] - last_close) / last_close
            result.ratio_1w = (median_30d[4] - last_close) / last_close
            result.ratio_2w = (median_30d[9] - last_close) / last_close
            result.ratio_3w = (median_30d[14] - last_close) / last_close
            result.ratio_4w = (median_30d[19] - last_close) / last_close

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
                result.ratio_quantile_multi[str(q)] = {
                    "1d": (q_values[0] - last_close) / last_close,
                    "1w": (q_values[4] - last_close) / last_close,
                    "2w": (q_values[9] - last_close) / last_close,
                    "3w": (q_values[14] - last_close) / last_close,
                    "4w": (q_values[19] - last_close) / last_close,
                }

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

        result.ratio_1d = result.ratio_30d[0]
        result.ratio_1w = result.ratio_30d[4]
        result.ratio_2w = result.ratio_30d[9]
        result.ratio_3w = result.ratio_30d[14]
        result.ratio_4w = result.ratio_30d[19]

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
            ratios = result.ratio_quantile_30d[str(q)]
            result.ratio_quantile_multi[str(q)] = {
                "1d": ratios[0],
                "1w": ratios[4],
                "2w": ratios[9],
                "3w": ratios[14],
                "4w": ratios[19],
            }

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

    def format_for_llm(self, forecast: TSFMForecast, format_type: int) -> str:
        """将预测结果格式化为LLM可读的字符串"""
        ticker = forecast.ticker

        # 定义解释文案
        expl_05 = "(0.05 indicates a 5%  probability of being less than this value)"
        expl_50 = "(0.5  indicates a 50% probability of being less than this value)"
        expl_95 = "(0.95 indicates a 95% probability of being less than this value)"

        if format_type == 1:
            # 格式1: 数字，30天
            prices = forecast.numeric_30d[:5] + ["..."] + forecast.numeric_30d[-5:]
            return f"TSFM Forecast for {ticker} (30-day price prediction):\n" \
                   f"Day 1-5: {[f'{p:.6f}' for p in forecast.numeric_30d[:5]]}\n" \
                   f"Day 26-30: {[f'{p:.6f}' for p in forecast.numeric_30d[-5:]]}"

        elif format_type == 2:
            # 格式2: 比例，30天
            return f"TSFM Forecast for {ticker} (30-day return prediction):\n" \
                   f"Day 1-5: {[f'{r * 100:.6f}%' for r in forecast.ratio_30d[:5]]}\n" \
                   f"Day 26-30: {[f'{r * 100:.6f}%' for r in forecast.ratio_30d[-5:]]}"

        elif format_type == 3:
            # 格式3: 比例，多时间窗口
            if forecast.ratio_1d is None:
                return f"TSFM Forecast for {ticker} (multi-horizon returns):\n" \
                       f"Status: {forecast.status}\n" \
                       f"Error: {forecast.error or 'No prediction available'}"
            return f"TSFM Forecast for {ticker} (multi-horizon returns):\n" \
                   f"1 Day: {forecast.ratio_1d * 100:.6f}%\n" \
                   f"1 Week: {forecast.ratio_1w * 100:.6f}%\n" \
                   f"2 Weeks: {forecast.ratio_2w * 100:.6f}%\n" \
                   f"3 Weeks: {forecast.ratio_3w * 100:.6f}%\n" \
                   f"4 Weeks: {forecast.ratio_4w * 100:.6f}%"

        elif format_type == 4:
            # 格式4: 数字，分位数，30天
            q50 = forecast.numeric_quantile_30d["0.5"]
            q05 = forecast.numeric_quantile_30d["0.05"]
            q95 = forecast.numeric_quantile_30d["0.95"]
            return f"TSFM Forecast for {ticker} (30-day quantile prices):\n" \
                   f"Median (50%): Day30=${q50[-1]:.6f} {expl_50}\n" \
                   f"5th percentile: Day30=${q05[-1]:.6f} {expl_05}\n" \
                   f"95th percentile: Day30=${q95[-1]:.6f} {expl_95}"

        elif format_type == 5:
            # 格式5: 比例，分位数，30天
            lines = [f"TSFM Forecast for {ticker} (30-day quantile returns):"]
            for q in ["0.05", "0.25", "0.5", "0.75", "0.95"]:
                r = forecast.ratio_quantile_30d[q][-1]
                note = ""
                if q == "0.05": note = f" {expl_05}"
                if q == "0.5": note = f" {expl_50}"
                if q == "0.95": note = f" {expl_95}"
                lines.append(f"  {q} quantile: {r * 100:.6f}%{note}")
            return "\n".join(lines)

        elif format_type == 6:
            # 格式6: 比例，分位数，多时间窗口
            if forecast.ratio_quantile_multi is None:
                return f"TSFM Forecast for {ticker} (quantile returns by horizon):\n" \
                       f"Status: {forecast.status}\n" \
                       f"Error: {forecast.error or 'No prediction available'}"
            lines = [f"TSFM Forecast for {ticker} (quantile returns by horizon):"]
            for horizon in ["1d", "1w", "2w", "3w", "4w"]:
                q05 = forecast.ratio_quantile_multi["0.05"][horizon]
                q50 = forecast.ratio_quantile_multi["0.5"][horizon]
                q95 = forecast.ratio_quantile_multi["0.95"][horizon]
                lines.append(
                    f"  {horizon}: [{q05 * 100:.1f}% {expl_05}, {q50 * 100:.1f}% {expl_50}, {q95 * 100:.1f}% {expl_95}]")
            return "\n".join(lines)

        elif format_type in (7, 8):
            # 格式7a/7b: 多时间窗口收益 + 过去7个已兑现1D预测的可靠性摘要
            if forecast.ratio_1d is None:
                return f"TSFM Forecast for {ticker} (multi-horizon returns + reliability):\n" \
                       f"Status: {forecast.status}\n" \
                       f"Error: {forecast.error or 'No prediction available'}"

            lines = [
                f"TSFM Forecast for {ticker} (multi-horizon returns):",
                f"1 Day: {forecast.ratio_1d * 100:.6f}%",
                f"1 Week: {forecast.ratio_1w * 100:.6f}%",
                f"2 Weeks: {forecast.ratio_2w * 100:.6f}%",
                f"3 Weeks: {forecast.ratio_3w * 100:.6f}%",
                f"4 Weeks: {forecast.ratio_4w * 100:.6f}%",
                "",
                f"TSFM Historical Reliability for {ticker} (computed from the last 7 resolved 1D forecasts before {forecast.forecast_date}):",
            ]

            reliability = (forecast.historical_reliability or {}).get("past_7_resolved_1d", {})
            if not reliability:
                lines.append("Insufficient historical reliability data.")
                return "\n".join(lines)

            n = int(reliability.get("n", 0))
            if n == 0:
                lines.append("Past 7 resolved 1D forecast MSE values: insufficient history")
                if format_type == 8:
                    lines.append("Normalized Reliability Score: insufficient history")
                lines.append("Sample Count: 0/7")
                return "\n".join(lines)

            window_size = int(reliability.get("window_size", 7))
            samples = reliability.get("samples", []) or []
            lines.append("Past 7 resolved 1D forecast MSE values (oldest to newest):")
            for sample in samples:
                origin_dt = sample.get("forecast_origin_date", "unknown")
                target_dt = sample.get("resolved_target_date", "unknown")
                sq_err = float(sample.get("squared_error", 0.0))
                lines.append(
                    f"  {origin_dt} -> {target_dt}: {sq_err * 10000:.4f} bp^2"
                )
            if format_type == 8:
                score = float(reliability.get("normalized_reliability_score", 0.0))
                lines.append(f"Normalized Reliability Score: {score:.3f}")
            lines.append(f"Sample Count: {n}/{window_size}")
            return "\n".join(lines)

        return f"Unknown format type: {format_type}"
