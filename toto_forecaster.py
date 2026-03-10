"""
Toto Forecaster - 封装 Toto 模型为与 Chronos 兼容的接口
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd

# 依赖：pip install toto torch numpy pandas
try:
    import torch
    from toto.data.util.dataset import MaskedTimeseries
    from toto.inference.forecaster import TotoForecaster
    from toto.model.toto import Toto
    _TOTO_AVAILABLE = True
except ImportError:
    torch = None
    MaskedTimeseries = None
    TotoForecaster = None
    Toto = None
    _TOTO_AVAILABLE = False


def _infer_freq_from_group(ts: pd.Series) -> pd.Timedelta:
    """
    从 timestamp 推断频率；推断失败则用中位数差分兜底。
    返回 pd.Timedelta，用于生成未来时间戳序列。
    """
    ts_sorted = ts.sort_values()
    # 1) pandas infer_freq
    freq_str = pd.infer_freq(ts_sorted)
    if freq_str is not None:
        try:
            offset = pd.tseries.frequencies.to_offset(freq_str)
            if hasattr(offset, 'delta'):
                return offset.delta
            test_date = pd.Timestamp('2000-01-01')
            next_date = test_date + offset
            return next_date - test_date
        except (AttributeError, TypeError):
            pass

    # 2) median delta fallback
    diffs = ts_sorted.diff().dropna()
    if len(diffs) == 0:
        return pd.Timedelta(days=1)

    med = diffs.median()
    if pd.isna(med) or med <= pd.Timedelta(0):
        return pd.Timedelta(days=1)
    return med


@dataclass
class TotoConfig:
    """Toto 模型配置"""
    # HuggingFace 上的模型 ID
    model_id: str = "Datadog/Toto-Open-Base-1.0"
    
    # 预测参数
    num_samples: int = 256  # 用于概率预测的样本数
    samples_per_batch: int = 256  # 控制推理时的内存使用
    
    # 是否编译模型以加速推理
    compile_model: bool = True
    
    # 设备
    device: Optional[str] = None  # None -> auto


class TotoForecasterWrapper:
    """
    将 Toto 封装成和 Chronos 相同的"predict_df"接口。
    """

    def __init__(self, cfg: Optional[TotoConfig] = None, device: Optional[str] = None):
        if not _TOTO_AVAILABLE:
            raise ImportError(
                "toto package not installed. Please install with: pip install toto"
            )
        
        self.cfg = cfg or TotoConfig()
        if device is not None:
            self.cfg.device = device
        self._toto_model = None
        self._forecaster = None

    def _lazy_init(self):
        """延迟初始化模型和预测器"""
        if self._forecaster is not None:
            return
        
        # device 自动选择
        if self.cfg.device is None:
            if torch.cuda.is_available():
                self.cfg.device = "cuda"
            else:
                self.cfg.device = "cpu"
        
        device_obj = torch.device(self.cfg.device)
        
        # 加载预训练模型
        self._toto_model = Toto.from_pretrained(self.cfg.model_id)
        self._toto_model.to(device_obj)
        
        # 可选：编译模型以加速推理
        if self.cfg.compile_model:
            try:
                self._toto_model.compile()
            except Exception as e:
                print(f"[WARNING] Failed to compile Toto model: {e}. Continuing without compilation.")
        
        # 创建预测器
        self._forecaster = TotoForecaster(self._toto_model.model)

    def predict_df(
        self,
        context_df: pd.DataFrame,
        future_df: Optional[pd.DataFrame] = None,
        prediction_length: int = 30,
        quantile_levels: Optional[Sequence[float]] = None,
        id_column: str = "id",
        timestamp_column: str = "timestamp",
        target: str = "target",
    ) -> pd.DataFrame:
        """
        输出 DataFrame：id, timestamp, predictions, [quantile columns...]
        
        参数:
            context_df: 上下文数据 DataFrame（包含 id, timestamp, target 列）
            future_df: 未来协变量（Toto 暂不支持，保留接口兼容性）
            prediction_length: 预测长度
            quantile_levels: 分位数列表，如 [0.05, 0.25, 0.5, 0.75, 0.95]
            id_column: ID 列名
            timestamp_column: 时间戳列名
            target: 目标值列名
        
        返回:
            预测结果 DataFrame
        """
        self._lazy_init()

        if quantile_levels is None:
            quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]

        # 兼容单序列：如果没有 id_column，就当作一个 id=0
        df = context_df.copy()
        if id_column not in df.columns:
            df[id_column] = 0

        # 按 id 分组，构造输入
        out_frames = []
        ids: List[Any] = []
        last_timestamps: List[pd.Timestamp] = []
        deltas: List[pd.Timedelta] = []

        for series_id, g in df.groupby(id_column):
            g2 = g.sort_values(timestamp_column)
            y = g2[target].to_numpy(dtype=np.float32)
            # 去除 NaN：简单线性插值 + 两端填充
            y = pd.Series(y).interpolate(limit_direction="both").to_numpy(dtype=np.float32)

            ts = pd.to_datetime(g2[timestamp_column])
            last_ts = ts.max()
            last_timestamps.append(last_ts)
            delta = _infer_freq_from_group(ts)
            deltas.append(delta)
            
            # Toto 需要将序列转换为 torch.Tensor
            # 输入格式：(channels, time_steps)
            # 对于单变量时间序列，channels=1
            input_series = torch.from_numpy(y).float().unsqueeze(0)  # (1, time_steps)
            input_series = input_series.to(self.cfg.device)
            
            # 准备时间戳信息（Toto API 期望但当前模型不使用）
            time_steps = len(y)
            timestamp_seconds = torch.zeros(1, time_steps).to(self.cfg.device)
            
            # 计算时间间隔（秒）
            # 如果 delta 是 Timedelta，转换为秒
            if isinstance(delta, pd.Timedelta):
                interval_seconds = int(delta.total_seconds())
            else:
                interval_seconds = 86400  # 默认1天（秒）
            
            time_interval_seconds = torch.tensor([interval_seconds], dtype=torch.float32).to(self.cfg.device)
            
            # 创建 MaskedTimeseries 对象
            inputs = MaskedTimeseries(
                series=input_series,
                padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
                id_mask=torch.zeros_like(input_series),
                timestamp_seconds=timestamp_seconds,
                time_interval_seconds=time_interval_seconds,
            )
            
            # 生成预测
            forecast = self._forecaster.forecast(
                inputs,
                prediction_length=prediction_length,
                num_samples=self.cfg.num_samples,
                samples_per_batch=self.cfg.samples_per_batch,
            )
            
            # 提取中位数预测（作为点预测）
            median_pred = forecast.median
            if hasattr(median_pred, 'cpu'):
                median_pred = median_pred.cpu().numpy()
            else:
                median_pred = np.array(median_pred)
            
            # 确保是1D数组
            if median_pred.ndim > 1:
                median_pred = median_pred.flatten()
            
            # 截取到 prediction_length
            median_pred = median_pred[:prediction_length]
            
            # 生成未来时间戳
            start = last_ts + delta
            idx = pd.date_range(start=start, periods=prediction_length, freq=delta)
            
            pred = pd.DataFrame(
                {
                    id_column: series_id,
                    timestamp_column: idx,
                    "predictions": median_pred,
                }
            )
            
            # 提取分位数预测
            for q in quantile_levels:
                try:
                    q_tensor = forecast.quantile(q)
                    if hasattr(q_tensor, 'cpu'):
                        q_value = q_tensor.cpu().numpy().flatten()[:prediction_length]
                    else:
                        q_value = np.array(q_tensor).flatten()[:prediction_length]
                    pred[str(q)] = q_value
                except Exception as e:
                    # 如果某个分位数提取失败，使用中位数作为替代
                    print(f"[Error] Failed to extract quantile {q} for series {series_id}: {e}")
                    raise e
            
            out_frames.append(pred)
            ids.append(series_id)

        pred_df = pd.concat(out_frames, ignore_index=True)
        return pred_df
