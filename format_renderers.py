from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict


def _normalize_quantile_key(q: float) -> str:
    return f"{float(q):g}"


def _anchor_quantile_keys(context: "RendererContext") -> tuple[str, str, str]:
    quantiles = sorted(float(q) for q in context.quantile_keys())
    lower_q = quantiles[0]
    upper_q = quantiles[-1]
    median_q = min(quantiles, key=lambda q: abs(q - 0.5))
    return (
        _normalize_quantile_key(lower_q),
        _normalize_quantile_key(median_q),
        _normalize_quantile_key(upper_q),
    )


def _to_percent(q_key: str) -> int:
    return int(round(float(q_key) * 100))


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


class RendererContext:
    """渲染 prompt 时需要共享的小型上下文。"""

    def __init__(
        self,
        *,
        horizon_specs,
        quantiles,
        quantile_keys: Callable[[], list[str]],
        quantile_explanations: Callable[[], tuple[str, str, str]],
    ):
        self.horizon_specs = horizon_specs
        self.quantiles = quantiles
        self.quantile_keys = quantile_keys
        self.quantile_explanations = quantile_explanations


class BaseFormatRenderer(ABC):
    """格式渲染抽象层，统一 render 接口。"""

    format_id: int

    @abstractmethod
    def render(self, forecast: Any, context: RendererContext) -> str:
        pass


class Format1Renderer(BaseFormatRenderer):
    format_id = 1

    def render(self, forecast: Any, context: RendererContext) -> str:
        # 格式1：数字，接下来30天
        ticker = forecast.ticker
        return f"TSFM Forecast for {ticker} (30-day price prediction):\n" \
               f"Day 1-5: {[f'{p:.6f}' for p in forecast.numeric_30d[:5]]}\n" \
               f"Day 26-30: {[f'{p:.6f}' for p in forecast.numeric_30d[-5:]]}"


class Format2Renderer(BaseFormatRenderer):
    format_id = 2

    def render(self, forecast: Any, context: RendererContext) -> str:
        # 格式2：比例，接下来30天
        ticker = forecast.ticker
        return f"TSFM Forecast for {ticker} (30-day return prediction):\n" \
               f"Day 1-5: {[f'{r * 100:.6f}%' for r in forecast.ratio_30d[:5]]}\n" \
               f"Day 26-30: {[f'{r * 100:.6f}%' for r in forecast.ratio_30d[-5:]]}"


class Format3Renderer(BaseFormatRenderer):
    format_id = 3

    def render(self, forecast: Any, context: RendererContext) -> str:
        # 格式3：比例，多时间窗口
        ticker = forecast.ticker
        if forecast.ratio_1d is None:
            return f"TSFM Forecast for {ticker} (multi-horizon returns):\n" \
                   f"Status: {forecast.status}\n" \
                   f"Error: {forecast.error or 'No prediction available'}"
        lines = [f"TSFM Forecast for {ticker} (multi-horizon returns):"]
        for spec in context.horizon_specs:
            ratio = getattr(forecast, spec.ratio_attr)
            lines.append(f"{spec.label}: {ratio * 100:.6f}%")
        return "\n".join(lines)


class Format4Renderer(BaseFormatRenderer):
    format_id = 4

    def render(self, forecast: Any, context: RendererContext) -> str:
        # 格式4：数字，分位数，30天
        ticker = forecast.ticker
        lower_key, median_key, upper_key = _anchor_quantile_keys(context)
        expl_lower, expl_median, expl_upper = context.quantile_explanations()
        median_pct = _to_percent(median_key)
        lower_pct = _to_percent(lower_key)
        upper_pct = _to_percent(upper_key)
        q50 = forecast.numeric_quantile_30d[median_key]
        q05 = forecast.numeric_quantile_30d[lower_key]
        q95 = forecast.numeric_quantile_30d[upper_key]
        return f"TSFM Forecast for {ticker} (30-day quantile prices):\n" \
               f"Median ({median_pct}%): Day30=${q50[-1]:.6f} {expl_median}\n" \
               f"{_ordinal(lower_pct)} percentile: Day30=${q05[-1]:.6f} {expl_lower}\n" \
               f"{_ordinal(upper_pct)} percentile: Day30=${q95[-1]:.6f} {expl_upper}"


class Format5Renderer(BaseFormatRenderer):
    format_id = 5

    def render(self, forecast: Any, context: RendererContext) -> str:
        # 格式5：比例，分位数，30天
        ticker = forecast.ticker
        lower_key, median_key, upper_key = _anchor_quantile_keys(context)
        expl_lower, expl_median, expl_upper = context.quantile_explanations()
        lines = [f"TSFM Forecast for {ticker} (30-day quantile returns):"]
        for q in context.quantile_keys():
            r = forecast.ratio_quantile_30d[q][-1]
            note = ""
            if q == lower_key:
                note = f" {expl_lower}"
            if q == median_key:
                note = f" {expl_median}"
            if q == upper_key:
                note = f" {expl_upper}"
            lines.append(f"  {q} quantile: {r * 100:.6f}%{note}")
        return "\n".join(lines)


class Format6Renderer(BaseFormatRenderer):
    format_id = 6

    def render(self, forecast: Any, context: RendererContext) -> str:
        # 格式6：比例，分位数，多时间窗口
        ticker = forecast.ticker
        lower_key, median_key, upper_key = _anchor_quantile_keys(context)
        expl_lower, expl_median, expl_upper = context.quantile_explanations()
        lower_pct = _to_percent(lower_key)
        median_pct = _to_percent(median_key)
        upper_pct = _to_percent(upper_key)
        if forecast.ratio_quantile_multi is None:
            return f"TSFM Forecast for {ticker} (quantile returns by horizon):\n" \
                   f"Status: {forecast.status}\n" \
                   f"Error: {forecast.error or 'No prediction available'}"
        lines = [f"TSFM Forecast for {ticker} (quantile returns by horizon):"]
        for spec in context.horizon_specs:
            horizon = spec.key
            q05 = forecast.ratio_quantile_multi[lower_key][horizon]
            q50 = forecast.ratio_quantile_multi[median_key][horizon]
            q95 = forecast.ratio_quantile_multi[upper_key][horizon]
            lines.append(
                f"  {horizon}: "
                f"[{_ordinal(lower_pct)}={q05 * 100:.1f}% {expl_lower}, "
                f"{median_pct}%={q50 * 100:.1f}% {expl_median}, "
                f"{_ordinal(upper_pct)}={q95 * 100:.1f}% {expl_upper}]"
            )
        return "\n".join(lines)


class BaseFormat7Renderer(BaseFormatRenderer):
    include_score: bool = False

    def render(self, forecast: Any, context: RendererContext) -> str:
        # 格式7a/7b：比例，多时间窗口 + 历史可靠性摘要
        ticker = forecast.ticker
        if forecast.ratio_1d is None:
            return f"TSFM Forecast for {ticker} (multi-horizon returns + reliability):\n" \
                   f"Status: {forecast.status}\n" \
                   f"Error: {forecast.error or 'No prediction available'}"

        lines = [f"TSFM Forecast for {ticker} (multi-horizon returns):"]
        for spec in context.horizon_specs:
            ratio = getattr(forecast, spec.ratio_attr)
            lines.append(f"{spec.label}: {ratio * 100:.6f}%")
        lines.extend([
            "",
            f"TSFM Historical Reliability for {ticker} (computed from the last 7 resolved 1D forecasts before {forecast.forecast_date}):",
        ])

        reliability = (forecast.historical_reliability or {}).get("past_7_resolved_1d", {})
        if not reliability:
            lines.append("Insufficient historical reliability data.")
            return "\n".join(lines)

        n = int(reliability.get("n", 0))
        if n == 0:
            lines.append("Past 7 resolved 1D forecast MSE values: insufficient history")
            if self.include_score:
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
        if self.include_score:
            score = float(reliability.get("normalized_reliability_score", 0.0))
            lines.append(f"Normalized Reliability Score: {score:.3f}")
        lines.append(f"Sample Count: {n}/{window_size}")
        return "\n".join(lines)


class Format7ARenderer(BaseFormat7Renderer):
    format_id = 7
    include_score = False


class Format7BRenderer(BaseFormat7Renderer):
    format_id = 8
    include_score = True


def build_format_renderers() -> Dict[int, BaseFormatRenderer]:
    renderers = [
        Format1Renderer(),
        Format2Renderer(),
        Format3Renderer(),
        Format4Renderer(),
        Format5Renderer(),
        Format6Renderer(),
        Format7ARenderer(),
        Format7BRenderer(),
    ]
    return {renderer.format_id: renderer for renderer in renderers}
