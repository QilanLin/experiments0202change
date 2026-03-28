"""
TimeFM Forecaster - 封装 TimeFM 模型为与 Chronos 兼容的接口
"""

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd

from .device_utils import select_timesfm_backend, select_torch_device

# 依赖：pip install timesfm torch numpy pandas
try:
    import torch
    import timesfm
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file as load_safetensors_file
    try:
        from transformers import TimesFmModelForPrediction
    except ImportError:
        TimesFmModelForPrediction = None
    _TIMESFM_AVAILABLE = True
except ImportError:
    torch = None
    timesfm = None
    snapshot_download = None
    load_safetensors_file = None
    TimesFmModelForPrediction = None
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


def _pad_array_left_to_multiple(values: np.ndarray, multiple: int) -> np.ndarray:
    """
    Left-pad a 1D array to a multiple of `multiple`.
    TimesFmModelForPrediction internally reshapes past_values by patch_length,
    so non-multiple context lengths (for example 231) will crash.
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {arr.shape}")
    if len(arr) == 0:
        return arr
    pad = (-len(arr)) % int(multiple)
    if pad == 0:
        return arr
    pad_values = np.full(pad, arr[0], dtype=arr.dtype)
    return np.concatenate([pad_values, arr])


def _validate_requested_quantiles_strict(
    quantile_levels: Sequence[float],
    supported: Sequence[float],
) -> list[float]:
    """Strictly validate that all requested quantiles are supported by TimesFM."""
    supported_rounded = [round(float(s), 2) for s in supported]
    requested_rounded = [round(float(q), 2) for q in quantile_levels]
    unsupported = sorted({q for q in requested_rounded if q not in supported_rounded})
    if unsupported:
        raise ValueError(
            "TimesFM strict quantile mode rejected unsupported quantiles. "
            f"requested={requested_rounded}, unsupported={unsupported}, "
            f"supported={supported_rounded}"
        )
    return requested_rounded


def _prepare_transformers_context_arrays(
    inputs: Sequence[np.ndarray],
    *,
    max_context_len: int,
    patch_length: int,
) -> tuple[list[np.ndarray], int]:
    """
    Clip to max_context_len, then pad each series to a patch-length multiple.
    Returns the padded arrays plus the max padded context length to feed into
    TimesFmModelForPrediction(... forecast_context_len=...).
    """
    prepared: list[np.ndarray] = []
    forecast_context_len = 0
    for ts in inputs:
        arr = np.asarray(ts, dtype=np.float32)
        if len(arr) > max_context_len:
            arr = arr[-max_context_len:]
        arr = _pad_array_left_to_multiple(arr, patch_length)
        prepared.append(arr)
        forecast_context_len = max(forecast_context_len, int(arr.shape[0]))
    return prepared, forecast_context_len


def _validate_legacy_timesfm_device(device: str) -> None:
    """
    The official TimesFM 2.5 torch implementation in the pinned upstream source
    only selects CUDA or CPU internally; it does not honor MPS.
    Since this repo forbids silent CPU fallback, reject non-CUDA devices here.
    """
    device_str = str(device)
    if device_str.startswith("cuda"):
        return
    raise RuntimeError(
        "TimesFM official 2.5 torch path requires CUDA in this repo. "
        f"Requested device={device_str!r}, but upstream torch code only selects "
        "CUDA or CPU internally. CPU fallback is disabled."
    )


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
    # 论文主实验默认只接受官方 2.5 torch API。
    # 如需临时调试 transformers fallback，必须显式打开。
    allow_transformers_fallback: bool = False

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
        # Prefer the official TimesFM 2.5 torch API when available. The
        # transformers TimesFM implementation does not currently load the
        # google/timesfm-2.5-200m-pytorch checkpoint cleanly in this setup.
        self._legacy_api = hasattr(timesfm, "TimesFM_2p5_200M_torch")
        self._use_transformers_model = (
            TimesFmModelForPrediction is not None
            and not self._legacy_api
            and self.cfg.allow_transformers_fallback
        )
        if not self._legacy_api and not self.cfg.allow_transformers_fallback:
            module_path = getattr(timesfm, "__file__", "unknown")
            raise RuntimeError(
                "TimesFM official 2.5 torch API is required for reproducible experiments, "
                "but the imported `timesfm` package does not expose `TimesFM_2p5_200M_torch`. "
                f"Imported module path: {module_path}. "
                "Use the official TimesFM clone/package on PYTHONPATH, or explicitly set "
                "`allow_transformers_fallback=True` only for non-paper debugging."
            )
        # TimesFM checkpoints expose a fixed quantile set (typically 0.1..0.9).
        # Keep strict semantics: unsupported requests should fail fast.
        self._supported_quantiles = tuple(round(i / 10, 1) for i in range(1, 10))

    def _patch_timesfm_torch_loader(self):
        """Make newer TimesFmTorch loaders accept HF safetensors checkpoints."""
        if self._legacy_api:
            return

        loader = timesfm.timesfm_torch.TimesFmTorch.load_from_checkpoint
        if getattr(loader, "_supports_safetensors", False):
            return

        def _patched_load_from_checkpoint(model, checkpoint):
            checkpoint_path = checkpoint.path
            repo_id = checkpoint.huggingface_repo_id
            if checkpoint_path is None:
                snapshot_dir = snapshot_download(
                    repo_id,
                    local_dir=checkpoint.local_dir,
                )
                checkpoint_path = os.path.join(snapshot_dir, "torch_model.ckpt")
                safetensors_path = os.path.join(snapshot_dir, "model.safetensors")
            else:
                safetensors_path = checkpoint_path.replace("torch_model.ckpt", "model.safetensors")

            model._model = timesfm.timesfm_torch.ppd.PatchedTimeSeriesDecoder(model._model_config)
            if os.path.exists(checkpoint_path):
                loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
            elif os.path.exists(safetensors_path):
                loaded_checkpoint = load_safetensors_file(safetensors_path)
            else:
                raise FileNotFoundError(
                    f"Neither torch_model.ckpt nor model.safetensors found for TimesFM checkpoint "
                    f"under repo {repo_id!r}"
                )

            model._model.load_state_dict(loaded_checkpoint)
            model._model.to(model._device)
            model._model.eval()

        _patched_load_from_checkpoint._supports_safetensors = True
        timesfm.timesfm_torch.TimesFmTorch.load_from_checkpoint = _patched_load_from_checkpoint

    def _lazy_init(self):
        if self._model is not None and self._compiled:
            return

        # device 自动选择
        if self.cfg.device is None:
            self.cfg.device = select_torch_device(torch_mod=torch)

        max_context = _round_up(int(self.cfg.max_context), 32)
        max_horizon = _round_up(int(self.cfg.max_horizon), 128)

        if self._use_transformers_model:
            self._model = TimesFmModelForPrediction.from_pretrained(self.cfg.model_id)
            self._model.to(self.cfg.device)
            self._model.eval()
            config_quantiles = getattr(self._model.config, "quantiles", None)
            if config_quantiles:
                self._supported_quantiles = tuple(round(float(q), 2) for q in config_quantiles)
        elif self._legacy_api:
            _validate_legacy_timesfm_device(self.cfg.device)
            # Older TimesFM package exposes the 2.5 torch checkpoint via a
            # convenience class plus ForecastConfig/compile.
            snapshot_dir = snapshot_download(
                self.cfg.model_id,
                local_files_only=True,
            )
            self._model = timesfm.TimesFM_2p5_200M_torch._from_pretrained(
                model_id=snapshot_dir,
                revision=None,
                cache_dir=None,
                force_download=False,
                local_files_only=True,
                token=None,
                config=None,
            )

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
        else:
            # timesfm>=1.3.0 exposes a unified TimesFm API that is initialized
            # with hparams + checkpoint instead of from_pretrained/compile.
            self._patch_timesfm_torch_loader()
            backend = select_timesfm_backend(self.cfg.device)
            hparams = timesfm.TimesFmHparams(
                context_len=max_context,
                horizon_len=max_horizon,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                num_heads=16,
                model_dims=1280,
                backend=backend,
                quantiles=self._supported_quantiles,
                point_forecast_mode="median",
            )
            checkpoint = timesfm.TimesFmCheckpoint(
                version="torch",
                huggingface_repo_id=self.cfg.model_id,
            )
            self._model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

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

        requested_quantiles = _validate_requested_quantiles_strict(
            quantile_levels,
            self._supported_quantiles,
        )
        supported_index = {
            round(float(supported_q), 2): idx
            for idx, supported_q in enumerate(self._supported_quantiles)
        }

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

        # 如果请求的 horizon 超出当前模型配置，重新初始化一个更大的 horizon。
        if prediction_length > int(self.cfg.max_horizon):
            new_cfg = TimesFMConfig(**{**self.cfg.__dict__})
            new_cfg.max_horizon = _round_up(prediction_length, 128)
            self.cfg = new_cfg
            self._model = None
            self._compiled = False
            self._lazy_init()

        max_context_len = _round_up(int(self.cfg.max_context), 32)

        if self._use_transformers_model:
            max_supported_horizon = int(getattr(self._model.config, "horizon_length", prediction_length))
            if prediction_length > max_supported_horizon:
                raise ValueError(
                    f"TimesFM transformers checkpoint only supports horizon_length <= {max_supported_horizon}; "
                    f"got {prediction_length}"
                )

            device = next(self._model.parameters()).device
            patch_length = int(getattr(self._model.config, "patch_length", 32))
            prepared_arrays, forecast_context_len = _prepare_transformers_context_arrays(
                inputs_list,
                max_context_len=max_context_len,
                patch_length=patch_length,
            )
            past_values = [
                torch.tensor(ts, dtype=torch.float32, device=device)
                for ts in prepared_arrays
            ]
            freq = torch.zeros(len(inputs_list), dtype=torch.long, device=device)
            with torch.no_grad():
                outputs = self._model(
                    past_values=past_values,
                    freq=freq,
                    forecast_context_len=forecast_context_len,
                )
            point_forecast = outputs.mean_predictions.detach().cpu().numpy()[:, :prediction_length]
            quantile_forecast = outputs.full_predictions.detach().cpu().numpy()[:, :prediction_length, :]
        elif self._legacy_api:
            point_forecast, quantile_forecast = self._model.forecast(
                horizon=int(prediction_length),
                inputs=inputs_list,
            )
        else:
            point_forecast, quantile_forecast = self._model.forecast(inputs=inputs_list)
            point_forecast = point_forecast[:, :prediction_length]
            quantile_forecast = quantile_forecast[:, :prediction_length, :]
        # quantile_forecast: (N, H, 10) -> index 0 is mean, 1..9 are quantiles
        # aligned with self._supported_quantiles.

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

            for q_ in requested_quantiles:
                q_idx = 1 + supported_index[q_]
                pred[f"{q_:.1f}"] = quantile_forecast[i, :, q_idx]

            out_frames.append(pred)

        pred_df = pd.concat(out_frames, ignore_index=True)
        return pred_df
