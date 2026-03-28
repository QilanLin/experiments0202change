"""
Experiment Configuration
"""
import os
from datetime import datetime
from typing import List, Dict, Any

from .format_registry import (
    FORMAT_SPEC_BY_CLI,
    HORIZON_SPECS,
    TSFM_QUANTILES,
    build_tsfm_output_formats_config,
)

# MAG7 股票列表
MAG7_TICKERS = ["AAPL", "GOOGL", "AMZN", "MSFT", "META", "TSLA", "NVDA"]

# 现金资产标识
CASH_TICKER = "CASH"

# 完整资产集合（MAG7 + CASH）
ASSET_TICKERS = MAG7_TICKERS + [CASH_TICKER]

# 实验配置
def _read_optional_positive_int_env(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    return None if value <= 0 else value


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


EXPERIMENT_CONFIG = {
    # 基本设置
    "initial_capital": 1_000_000,  # 本金 1M
    "simulation_days": 30,  # 模拟周期
    "tickers": ASSET_TICKERS,  # 包含 CASH
    
    # LLM 配置
    "debug_llm": os.getenv("QWEN_DEBUG_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
    "production_llm": os.getenv("QWEN_PRODUCTION_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
    "llm_provider": "qwen",
    "llm_temperature": _read_float_env("LLM_TEMPERATURE", 0.0),
    "llm_max_new_tokens": _read_optional_positive_int_env("LLM_MAX_NEW_TOKENS", 10240),
    "lmstudio_base_url": os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
    "lmstudio_api_key": os.getenv("LM_STUDIO_API_KEY"),
    "llm_input_token_budget": _read_optional_positive_int_env("LLM_INPUT_TOKEN_BUDGET", 240000),
    
    # TSFM 配置
    "tsfm_model": "amazon/chronos-2",
    "tsfm_lookback_days": 365,  # 过去一年股价
    "tsfm_prediction_horizons": [spec.days for spec in HORIZON_SPECS],  # 1天, 1周, 2周, 3周, 4周
    "tsfm_quantiles": list(TSFM_QUANTILES),
    
    # 输出格式配置（具体格式定义与中文说明统一收敛到 format_registry.py）
    "tsfm_output_formats": build_tsfm_output_formats_config(),
    "tsfm_reliability_window_size": 7,
    "tsfm_reliability_metrics": [
        "past_7_resolved_1d_mse",
        "normalized_reliability_score",
    ],
    
    # 数据源
    "data_vendor": "alpha_vantage",
    
    # 结果保存
    "results_dir": "./experiment_results",
    "save_intermediate": True,
}

# 实验类型
class ExperimentType:
    BASELINE_LLM_ONLY = FORMAT_SPEC_BY_CLI["baseline"].experiment_type
    LLM_TSFM_FORMAT_1 = FORMAT_SPEC_BY_CLI["tsfm_1"].experiment_type
    LLM_TSFM_FORMAT_2 = FORMAT_SPEC_BY_CLI["tsfm_2"].experiment_type
    LLM_TSFM_FORMAT_3 = FORMAT_SPEC_BY_CLI["tsfm_3"].experiment_type
    LLM_TSFM_FORMAT_4 = FORMAT_SPEC_BY_CLI["tsfm_4"].experiment_type
    LLM_TSFM_FORMAT_5 = FORMAT_SPEC_BY_CLI["tsfm_5"].experiment_type
    LLM_TSFM_FORMAT_6 = FORMAT_SPEC_BY_CLI["tsfm_6"].experiment_type
    LLM_TSFM_FORMAT_7A = FORMAT_SPEC_BY_CLI["tsfm_7a"].experiment_type
    LLM_TSFM_FORMAT_7B = FORMAT_SPEC_BY_CLI["tsfm_7b"].experiment_type


def get_experiment_dir(experiment_type: str, run_id: str = None) -> str:
    """获取实验结果目录"""
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = EXPERIMENT_CONFIG["results_dir"]
    return os.path.join(base_dir, experiment_type, run_id)
