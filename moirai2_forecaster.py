"""
Moirai2 Forecaster - 封装 Uni2TS Moirai2 模型为与 Chronos 兼容的接口
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd

# 依赖：pip install uni2ts torch numpy pandas gluonts
try:
    import torch
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.field_names import FieldName
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    _MOIRAI2_AVAILABLE = True
except ImportError:
    torch = None
    ListDataset = None
    FieldName = None
    Moirai2Forecast = None
    Moirai2Module = None
    _MOIRAI2_AVAILABLE = False


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
class Moirai2Config:
    """Moirai2 模型配置"""
    # HuggingFace 上的模型 ID
    model_id: str = "Salesforce/moirai-2.0-R-small"
    
    # 上下文长度（根据预测长度自动调整）
    context_length: int = 1680
    
    # 预测长度（默认30天）
    prediction_length: int = 100
    
    # 批次大小
    batch_size: int = 32
    
    # 设备
    device: Optional[str] = None  # None -> auto


class Moirai2Forecaster:
    """
    将 Moirai2 封装成和 Chronos 相同的"predict_df"接口。
    """

    def __init__(self, cfg: Optional[Moirai2Config] = None, device: Optional[str] = None):
        if not _MOIRAI2_AVAILABLE:
            raise ImportError(
                "uni2ts package not installed. Please install with: pip install uni2ts"
            )
        
        self.cfg = cfg or Moirai2Config()
        if device is not None:
            self.cfg.device = device
        self._model = None
        self._predictor = None

    def _lazy_init(self, prediction_length: int = 30):
        """延迟初始化模型和预测器"""
        if self._predictor is not None:
            # 如果预测长度没有变化，直接返回
            if prediction_length <= self.cfg.prediction_length:
                return
        
        # 根据预测长度调整 context_length（Moirai2 建议 context_length 至少是 prediction_length 的几倍）
        # 默认使用 1680，如果预测长度较大，则相应增加
        context_length = max(self.cfg.context_length, prediction_length * 10)
        
        # device 自动选择
        if self.cfg.device is None:
            if torch.cuda.is_available():
                self.cfg.device = "cuda"
            else:
                self.cfg.device = "cpu"

        # 加载模型
        module = Moirai2Module.from_pretrained(self.cfg.model_id)
        
        # 创建 Moirai2Forecast 实例
        self._model = Moirai2Forecast(
            module=module,
            prediction_length=prediction_length,
            context_length=context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        
        # 创建预测器
        self._predictor = self._model.create_predictor(batch_size=self.cfg.batch_size)
        
        # 更新配置中的预测长度
        self.cfg.prediction_length = prediction_length
        self.cfg.context_length = context_length

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
            future_df: 未来协变量（Moirai2 暂不支持，保留接口兼容性）
            prediction_length: 预测长度
            quantile_levels: 分位数列表，如 [0.05, 0.25, 0.5, 0.75, 0.95]
            id_column: ID 列名
            timestamp_column: 时间戳列名
            target: 目标值列名
        
        返回:
            预测结果 DataFrame
        """
        self._lazy_init(prediction_length=prediction_length)

        if quantile_levels is None:
            quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]

        # 兼容单序列：如果没有 id_column，就当作一个 id=0
        df = context_df.copy()
        if id_column not in df.columns:
            df[id_column] = 0

        # 按 id 分组，构造 GluonTS ListDataset
        dataset_list = []
        ids: List[Any] = []
        last_timestamps: List[pd.Timestamp] = []
        deltas: List[pd.Timedelta] = []
        freq_str = "D"  # 默认频率

        for series_id, g in df.groupby(id_column):
            g2 = g.sort_values(timestamp_column)
            y = g2[target].to_numpy(dtype=np.float32)
            # 去除 NaN：简单线性插值 + 两端填充
            y = pd.Series(y).interpolate(limit_direction="both").to_numpy(dtype=np.float32)

            ts = pd.to_datetime(g2[timestamp_column])
            first_ts = ts.min()  # 使用第一个时间戳作为 START
            last_ts = ts.max()
            last_timestamps.append(last_ts)
            delta = _infer_freq_from_group(ts)
            deltas.append(delta)

            # 构造 GluonTS 格式的数据
            # Moirai2 需要 start 和 target 字段
            # START 应该是序列的第一个时间戳
            # 推断频率字符串（用于 ListDataset）- 只在第一次推断
            if len(dataset_list) == 0:
                inferred_freq = pd.infer_freq(ts)
                if inferred_freq is None:
                    # 如果无法推断，使用中位数差分来推断
                    # 将 Timedelta 转换为频率字符串
                    if delta == pd.Timedelta(days=1):
                        freq_str = "D"
                    elif delta == pd.Timedelta(weeks=1) or delta == pd.Timedelta(days=7):
                        freq_str = "W"
                    elif delta >= pd.Timedelta(days=28) and delta <= pd.Timedelta(days=31):
                        freq_str = "M"
                    else:
                        freq_str = "D"  # 默认日频
                else:
                    # 简化频率字符串（移除偏移量等）
                    if "D" in inferred_freq or "day" in inferred_freq.lower():
                        freq_str = "D"
                    elif "W" in inferred_freq or "week" in inferred_freq.lower():
                        freq_str = "W"
                    elif "M" in inferred_freq or "month" in inferred_freq.lower():
                        freq_str = "M"
                    else:
                        freq_str = "D"  # 默认日频
            
            dataset_entry = {
                FieldName.START: pd.Timestamp(first_ts),
                FieldName.TARGET: y,
            }
            dataset_list.append(dataset_entry)
            ids.append(series_id)

        # 创建 ListDataset（使用推断的频率）
        dataset = ListDataset(dataset_list, freq=freq_str)

        # 调用预测器
        forecasts = list(self._predictor.predict(dataset))

        # 组装输出 pred_df
        out_frames = []
        for i, (series_id, forecast) in enumerate(zip(ids, forecasts)):
            # 生成未来时间戳
            start = last_timestamps[i] + deltas[i]
            idx = pd.date_range(start=start, periods=prediction_length, freq=deltas[i])

            # 提取均值预测（median/mean）
            # Moirai2 的 Forecast 对象有 mean 属性
            # mean 可能是 torch.Tensor 或 numpy array
            mean_tensor = forecast.mean
            if hasattr(mean_tensor, 'numpy'):
                mean_pred = mean_tensor.numpy().flatten()[:prediction_length]
            elif hasattr(mean_tensor, 'cpu'):
                mean_pred = mean_tensor.cpu().numpy().flatten()[:prediction_length]
            else:
                mean_pred = np.array(mean_tensor).flatten()[:prediction_length]
            
            pred = pd.DataFrame(
                {
                    id_column: series_id,
                    timestamp_column: idx,
                    "predictions": mean_pred,
                }
            )

            # 提取分位数预测
            # Moirai2 的 Forecast 对象有 quantile 方法
            for q in quantile_levels:
                try:
                    q_tensor = forecast.quantile(q)
                    # 处理不同的 tensor 类型
                    if hasattr(q_tensor, 'numpy'):
                        q_value = q_tensor.numpy().flatten()[:prediction_length]
                    elif hasattr(q_tensor, 'cpu'):
                        q_value = q_tensor.cpu().numpy().flatten()[:prediction_length]
                    else:
                        q_value = np.array(q_tensor).flatten()[:prediction_length]
                    pred[str(q)] = q_value
                except Exception as e:
                    # 如果某个分位数提取失败，使用均值作为替代
                    print(f"[WARNING] Failed to extract quantile {q} for series {series_id}: {e}")
                    pred[str(q)] = mean_pred

            out_frames.append(pred)

        pred_df = pd.concat(out_frames, ignore_index=True)
        return pred_df
