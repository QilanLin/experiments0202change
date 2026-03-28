from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

import pandas as pd

from .device_utils import select_torch_device

try:
    from chronos import Chronos2Pipeline
    _CHRONOS_AVAILABLE = True
except ImportError:
    Chronos2Pipeline = None
    _CHRONOS_AVAILABLE = False

try:
    from .timesfm_forecaster import TimesFMForecaster, TimesFMConfig
    _TIMESFM_AVAILABLE = True
except ImportError:
    TimesFMForecaster = None
    TimesFMConfig = None
    _TIMESFM_AVAILABLE = False

try:
    from .moirai2_forecaster import Moirai2Forecaster, Moirai2Config
    _MOIRAI2_AVAILABLE = True
except ImportError:
    Moirai2Forecaster = None
    Moirai2Config = None
    _MOIRAI2_AVAILABLE = False

try:
    from .toto_forecaster import TotoForecasterWrapper, TotoConfig
    _TOTO_AVAILABLE = True
except ImportError:
    TotoForecasterWrapper = None
    TotoConfig = None
    _TOTO_AVAILABLE = False


_CHRONOS_PIPELINE = None


class BaseTSFMBackend(ABC):
    """TSFM backend 抽象层，统一 pipeline 加载和 predict_df 接口。"""

    backend_name: str = "base"

    def __init__(self, *, device: str | None = None, model_name: str | None = None):
        self.device = device
        self.model_name = model_name
        self._pipeline = None

    @abstractmethod
    def ensure_available(self) -> None:
        """在 backend 不可用时抛出带上下文的 ImportError。"""

    @abstractmethod
    def _build_pipeline(self) -> Any:
        """构造 backend 专属 pipeline。"""

    @abstractmethod
    def predict_df(
        self,
        context_df: pd.DataFrame,
        prediction_length: int,
        quantile_levels: List[float],
    ) -> pd.DataFrame:
        """统一的预测接口。"""

    def load_pipeline(self) -> Any:
        self.ensure_available()
        if self._pipeline is None:
            self._pipeline = self._build_pipeline()
        return self._pipeline


class ChronosBackend(BaseTSFMBackend):
    backend_name = "chronos"

    def ensure_available(self) -> None:
        if not _CHRONOS_AVAILABLE:
            raise ImportError("chronos-forecasting not installed")

    def _build_pipeline(self) -> Any:
        global _CHRONOS_PIPELINE

        if _CHRONOS_PIPELINE is None:
            try:
                import torch
                device_map = select_torch_device(self.device, torch_mod=torch)
            except ImportError:
                device_map = "cpu"

            # 保持原有行为：Chronos backend 复用一个全局 pipeline，避免重复加载大模型。
            _CHRONOS_PIPELINE = Chronos2Pipeline.from_pretrained(
                self.model_name or "amazon/chronos-2", device_map=device_map
            )

        return _CHRONOS_PIPELINE

    def predict_df(
        self,
        context_df: pd.DataFrame,
        prediction_length: int,
        quantile_levels: List[float],
    ) -> pd.DataFrame:
        pipeline = self.load_pipeline()
        return pipeline.predict_df(
            context_df,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )


class TimesFMBackend(BaseTSFMBackend):
    backend_name = "timesfm"

    def ensure_available(self) -> None:
        if not _TIMESFM_AVAILABLE:
            raise ImportError(
                "TimeFM forecaster is not available. Please install timesfm package: pip install timesfm"
            )

    def _build_pipeline(self) -> Any:
        cfg = TimesFMConfig(device=self.device)
        return TimesFMForecaster(cfg=cfg, device=self.device)

    def predict_df(
        self,
        context_df: pd.DataFrame,
        prediction_length: int,
        quantile_levels: List[float],
    ) -> pd.DataFrame:
        # TimeFM wrapper 内部使用严格 quantile 校验：不支持的 quantile 会直接报错。
        pipeline = self.load_pipeline()
        return pipeline.predict_df(
            context_df,
            future_df=None,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )


class Moirai2Backend(BaseTSFMBackend):
    backend_name = "moirai2"

    def ensure_available(self) -> None:
        if not _MOIRAI2_AVAILABLE:
            raise ImportError(
                "Moirai2 forecaster is not available. Please install uni2ts package: pip install uni2ts"
            )

    def _build_pipeline(self) -> Any:
        cfg = Moirai2Config(device=self.device)
        return Moirai2Forecaster(cfg=cfg, device=self.device)

    def predict_df(
        self,
        context_df: pd.DataFrame,
        prediction_length: int,
        quantile_levels: List[float],
    ) -> pd.DataFrame:
        pipeline = self.load_pipeline()
        return pipeline.predict_df(
            context_df,
            future_df=None,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )


class TotoBackend(BaseTSFMBackend):
    backend_name = "toto"

    def ensure_available(self) -> None:
        if not _TOTO_AVAILABLE:
            raise ImportError(
                "Toto forecaster is not available. Please install toto package: pip install toto"
            )

    def _build_pipeline(self) -> Any:
        cfg = TotoConfig(device=self.device)
        return TotoForecasterWrapper(cfg=cfg, device=self.device)

    def predict_df(
        self,
        context_df: pd.DataFrame,
        prediction_length: int,
        quantile_levels: List[float],
    ) -> pd.DataFrame:
        pipeline = self.load_pipeline()
        return pipeline.predict_df(
            context_df,
            future_df=None,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )


BACKEND_CLASSES = {
    "chronos": ChronosBackend,
    "timesfm": TimesFMBackend,
    "moirai2": Moirai2Backend,
    "toto": TotoBackend,
}


def build_tsfm_backend(
    backend_name: str,
    *,
    device: str | None = None,
    model_name: str | None = None,
) -> BaseTSFMBackend:
    """统一 backend 工厂：外部仍传字符串，内部返回多态 adapter。"""
    backend_cls = BACKEND_CLASSES.get(backend_name)
    if backend_cls is None:
        raise ValueError(
            f"Invalid forecaster_type: {backend_name}. "
            f"Supported values: {', '.join(repr(name) for name in BACKEND_CLASSES)}"
        )
    backend = backend_cls(device=device, model_name=model_name)
    backend.ensure_available()
    return backend
