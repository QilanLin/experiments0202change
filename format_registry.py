from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class HorizonSpec:
    key: str
    days: int
    label: str
    ratio_attr: str


@dataclass(frozen=True)
class FormatSpec:
    cli_name: str
    experiment_type: str
    format_id: Optional[int]
    config_key: Optional[str]
    config_value: Optional[str]
    display_name: str
    description: str

    @property
    def uses_tsfm(self) -> bool:
        return self.format_id is not None


# 共享的多时间窗口定义（1天, 1周, 2周, 3周, 4周）
HORIZON_SPECS: tuple[HorizonSpec, ...] = (
    HorizonSpec(key="1d", days=1, label="1 Day", ratio_attr="ratio_1d"),
    HorizonSpec(key="1w", days=5, label="1 Week", ratio_attr="ratio_1w"),
    HorizonSpec(key="2w", days=10, label="2 Weeks", ratio_attr="ratio_2w"),
    HorizonSpec(key="3w", days=15, label="3 Weeks", ratio_attr="ratio_3w"),
    HorizonSpec(key="4w", days=20, label="4 Weeks", ratio_attr="ratio_4w"),
)

# 共享的分位数定义
TSFM_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)

FORMAT_SPECS: tuple[FormatSpec, ...] = (
    # Baseline：LLM Only
    FormatSpec(
        cli_name="baseline",
        experiment_type="baseline_llm_only",
        format_id=None,
        config_key=None,
        config_value=None,
        display_name="Baseline (LLM Only)",
        description="LLM only, no TSFM input",
    ),
    # 格式1：数字，接下来30天
    FormatSpec(
        cli_name="tsfm_1",
        experiment_type="llm_tsfm_format_1",
        format_id=1,
        config_key="format_1",
        config_value="numeric_30d",
        display_name="TSFM Format 1 (Numeric 30d)",
        description="30-day price forecast, showing Day 1-5 and Day 26-30",
    ),
    # 格式2：比例，接下来30天
    FormatSpec(
        cli_name="tsfm_2",
        experiment_type="llm_tsfm_format_2",
        format_id=2,
        config_key="format_2",
        config_value="ratio_30d",
        display_name="TSFM Format 2 (Ratio 30d)",
        description="30-day return forecast, showing Day 1-5 and Day 26-30",
    ),
    # 格式3：比例，1天/1周/2周/3周/4周
    FormatSpec(
        cli_name="tsfm_3",
        experiment_type="llm_tsfm_format_3",
        format_id=3,
        config_key="format_3",
        config_value="ratio_multi_horizon",
        display_name="TSFM Format 3 (Ratio Multi-Horizon)",
        description="Multi-horizon returns at 1d, 1w, 2w, 3w, 4w",
    ),
    # 格式4：数字，分位数，30天
    FormatSpec(
        cli_name="tsfm_4",
        experiment_type="llm_tsfm_format_4",
        format_id=4,
        config_key="format_4",
        config_value="numeric_quantile_30d",
        display_name="TSFM Format 4 (Numeric Quantile 30d)",
        description="Day-30 price quantiles",
    ),
    # 格式5：比例，分位数，30天
    FormatSpec(
        cli_name="tsfm_5",
        experiment_type="llm_tsfm_format_5",
        format_id=5,
        config_key="format_5",
        config_value="ratio_quantile_30d",
        display_name="TSFM Format 5 (Ratio Quantile 30d)",
        description="Day-30 return quantiles",
    ),
    # 格式6：比例，分位数，多时间窗口
    FormatSpec(
        cli_name="tsfm_6",
        experiment_type="llm_tsfm_format_6",
        format_id=6,
        config_key="format_6",
        config_value="ratio_quantile_multi",
        display_name="TSFM Format 6 (Ratio Quantile Multi-Horizon)",
        description="Multi-horizon return quantiles",
    ),
    # 格式7a：比例，多时间窗口 + 过去7个已兑现1D预测的逐条MSE值
    FormatSpec(
        cli_name="tsfm_7a",
        experiment_type="llm_tsfm_format_7a",
        format_id=7,
        config_key="format_7a",
        config_value="ratio_multi_horizon_with_mse",
        display_name="TSFM Format 7a (Seven 1D MSE Values)",
        description="Multi-horizon returns plus seven individual resolved 1D MSE values",
    ),
    # 格式7b：格式7a + 归一化可靠性分数
    FormatSpec(
        cli_name="tsfm_7b",
        experiment_type="llm_tsfm_format_7b",
        format_id=8,
        config_key="format_7b",
        config_value="ratio_multi_horizon_with_mse_and_score",
        display_name="TSFM Format 7b (Seven 1D MSE Values + Score)",
        description="Format 7a plus normalized reliability score",
    ),
)

FORMAT_SPEC_BY_CLI = {spec.cli_name: spec for spec in FORMAT_SPECS}
FORMAT_SPEC_BY_EXPERIMENT_TYPE = {spec.experiment_type: spec for spec in FORMAT_SPECS}
CLI_EXPERIMENT_CHOICES = tuple(spec.cli_name for spec in FORMAT_SPECS)
TSFM_FORMAT_SPECS = tuple(spec for spec in FORMAT_SPECS if spec.uses_tsfm)
TSFM_FORMAT_IDS = tuple(spec.format_id for spec in TSFM_FORMAT_SPECS if spec.format_id is not None)


def build_tsfm_output_formats_config() -> dict[str, str]:
    return {
        spec.config_key: spec.config_value
        for spec in FORMAT_SPECS
        if spec.config_key is not None and spec.config_value is not None
    }


def build_experiment_entries(
    *,
    include_baseline: bool = True,
    format_ids: Optional[Iterable[int]] = None,
) -> list[dict[str, object]]:
    """基于统一 registry 生成实验列表，避免批跑脚本各自维护一份。"""
    selected_ids = None if format_ids is None else set(format_ids)
    experiments: list[dict[str, object]] = []
    for spec in FORMAT_SPECS:
        if not include_baseline and not spec.uses_tsfm:
            continue
        if spec.uses_tsfm and selected_ids is not None and spec.format_id not in selected_ids:
            continue
        experiments.append(
            {
                "type": spec.experiment_type,
                "tsfm_format": spec.format_id,
                "name": spec.display_name,
                "cli_name": spec.cli_name,
            }
        )
    return experiments
