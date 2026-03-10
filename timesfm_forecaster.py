"""
TimeFM Forecaster - 封装 TimeFM 模型为与 Chronos 兼容的接口
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd

# 依赖：pip install timesfm torch numpy pandas
try:
    import torch
    import timesfm
    _TIMESFM_AVAILABLE = True
except ImportError:
    torch = None
    timesfm = None
    _TIMESFM_AVAILABLE = False


def _infer_freq_from_group(ts: pd.Series) -> pd.Timedelta:
    """
    尽量从 timestamp 推断频率；推断失败则用中位数差分兜底。
    返回 pd.Timedelta，用于生成未来时间戳序列。
    """
    ts_sorted = ts.sort_values()
    # 1) pandas infer_freq
    freq_str = pd.infer_freq(ts_sorted)
    if freq_str is not None:
        try:
            offset = pd.tseries.frequencies.to_offset(freq_str)
            # 尝试获取 delta 属性（某些频率对象可能没有）
            if hasattr(offset, 'delta'):
                return offset.delta
            # 如果没有 delta 属性，尝试通过计算得到
            # 创建一个测试日期，应用 offset，然后计算差值
            test_date = pd.Timestamp('2000-01-01')
            next_date = test_date + offset
            return next_date - test_date
        except (AttributeError, TypeError):
            # 如果获取失败，继续使用中位数差分
            pass

    # 2) median delta fallback
    diffs = ts_sorted.diff().dropna()
    if len(diffs) == 0:
        # 极端：只有一个点，兜底为 1 天
        return pd.Timedelta(days=1)

    # diffs 可能是 Timedelta 类型
    med = diffs.median()
    if pd.isna(med) or med <= pd.Timedelta(0):
        return pd.Timedelta(days=1)
    return med


def _round_up(x: int, base: int) -> int:
    return int(math.ceil(x / base) * base)


@dataclass
class TimesFMConfig:
    # HuggingFace 上的官方 checkpoint
    model_id: str = "google/timesfm-2.5-200m-pytorch"

    # compile 用的最大上下文/最大预测范围（会自动向上取整以满足 patch 约束）
    max_context: int = 1024
    max_horizon: int = 256

    # forecast flags（与 timesfm.ForecastConfig 对齐）
    normalize_inputs: bool = True
    use_continuous_quantile_head: bool = True
    force_flip_invariance: bool = True
    infer_is_positive: bool = True
    fix_quantile_crossing: bool = True
    return_backcast: bool = False

    # 设备
    device: Optional[str] = None  # None -> auto


class TimesFMForecaster:
    """
    将 TimesFM 封装成和 Chronos 相同的"predict_df"接口。
    注意：TimesFM 2.5 的 forecast(inputs=[np.array(...), ...]) 只需要数值序列，
    这里先忽略 future_df 的协变量（保持接口一致，后续需要再扩展 xreg）。
    """

    def __init__(self, cfg: Optional[TimesFMConfig] = None, device: Optional[str] = None):
        if not _TIMESFM_AVAILABLE:
            raise ImportError("timesfm package not installed. Please install with: pip install timesfm")
        
        self.cfg = cfg or TimesFMConfig()
        if device is not None:
            self.cfg.device = device
        self._model = None
        self._compiled = False

    def _lazy_init(self):
        if self._model is not None and self._compiled:
            return

        # device 自动选择
        if self.cfg.device is None:
            if torch.cuda.is_available():
                self.cfg.device = "cuda"
            else:
                self.cfg.device = "cpu"

        # 加载模型
        # TimesFM 官方示例：timesfm.TimesFM_2p5_200M_torch.from_pretrained(...)
        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.cfg.model_id)

        # TimesFM 2.5 的 patch 约束：context 通常需对齐 patch_len(32)，horizon 对齐 output_patch_len(128)；
        # 其 compile 内部也会自动向上取整，但我们提前做一层保守处理，减少意外。
        # 这里用保守基数：context_base=32, horizon_base=128
        max_context = _round_up(int(self.cfg.max_context), 32)
        max_horizon = _round_up(int(self.cfg.max_horizon), 128)

        fc = timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=max_horizon,
            normalize_inputs=self.cfg.normalize_inputs,
            use_continuous_quantile_head=self.cfg.use_continuous_quantile_head,
            force_flip_invariance=self.cfg.force_flip_invariance,
            infer_is_positive=self.cfg.infer_is_positive,
            fix_quantile_crossing=self.cfg.fix_quantile_crossing,
            return_backcast=self.cfg.return_backcast,
        )

        self._model.compile(fc)
        self._compiled = True

    def predict_df(
        self,
        context_df: pd.DataFrame,
        future_df: Optional[pd.DataFrame] = None,
        prediction_length: int = 24,
        quantile_levels: Optional[Sequence[float]] = None,
        id_column: str = "id",
        timestamp_column: str = "timestamp",
        target: str = "target",
    ) -> pd.DataFrame:
        """
        输出 DataFrame：id, timestamp, predictions, [quantile columns...]
        quantile_levels 建议使用 {0.1,0.2,...,0.9} 子集；常用 {0.1,0.5,0.9}。
        """
        self._lazy_init()

        if quantile_levels is None:
            quantile_levels = [0.1, 0.5, 0.9]

        # TimesFM 的 quantile_forecast shape: (N, horizon, 10)
        # index 0 是 mean；index 1..9 对应 0.1..0.9
        supported = {round(i / 10, 1) for i in range(1, 10)}
        for q in quantile_levels:
            q_ = float(q)
            q_ = round(q_, 1)
            if q_ not in supported:
                raise ValueError(f"TimesFM only supports quantiles in {sorted(supported)}; got {q}")

        # 兼容单序列：如果没有 id_column，就当作一个 id=0
        df = context_df.copy()
        if id_column not in df.columns:
            df[id_column] = 0

        # 按 id 分组，构造 inputs_list
        inputs_list: List[np.ndarray] = []
        ids: List[Any] = []
        last_timestamps: List[pd.Timestamp] = []
        deltas: List[pd.Timedelta] = []

        for series_id, g in df.groupby(id_column):
            g2 = g.sort_values(timestamp_column)
            y = g2[target].to_numpy(dtype=np.float32)
            # 去除 NaN：简单线性插值 + 两端填充
            y = pd.Series(y).interpolate(limit_direction="both").to_numpy(dtype=np.float32)

            inputs_list.append(y)
            ids.append(series_id)

            ts = pd.to_datetime(g2[timestamp_column])
            last_ts = ts.max()
            last_timestamps.append(last_ts)
            deltas.append(_infer_freq_from_group(ts))

        # 确保 compile 覆盖 prediction_length（如果超出 max_horizon，则重新 compile 更大的 max_horizon）
        if prediction_length > int(self._model.forecast_config.max_horizon):
            # 重新 compile（增量：仅在需要时发生）
            new_cfg = TimesFMConfig(**{**self.cfg.__dict__})
            new_cfg.max_horizon = _round_up(prediction_length, 128)
            self.cfg = new_cfg
            self._model = None
            self._compiled = False
            self._lazy_init()

        point_forecast, quantile_forecast = self._model.forecast(
            horizon=int(prediction_length),
            inputs=inputs_list,
        )
        # point_forecast: (N, H)
        # quantile_forecast: (N, H, 10)

        # 组装输出 pred_df
        out_frames = []
        for i, series_id in enumerate(ids):
            start = last_timestamps[i] + deltas[i]
            idx = pd.date_range(start=start, periods=prediction_length, freq=deltas[i])

            pred = pd.DataFrame(
                {
                    id_column: series_id,
                    timestamp_column: idx,
                    "predictions": point_forecast[i, :],
                }
            )

            for q in quantile_levels:
                q_ = round(float(q), 1)
                q_idx = int(q_ * 10)  # 0.1->1, 0.5->5, 0.9->9
                pred[f"{q_:.1f}"] = quantile_forecast[i, :, q_idx]

            out_frames.append(pred)

        pred_df = pd.concat(out_frames, ignore_index=True)
        return pred_df
