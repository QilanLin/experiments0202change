"""
Experiment Configuration
"""
import os
from datetime import datetime
from typing import List, Dict, Any

# MAG7 股票列表
MAG7_TICKERS = ["AAPL", "GOOGL", "AMZN", "MSFT", "META", "TSLA", "NVDA"]

# 现金资产标识
CASH_TICKER = "CASH"

# 完整资产集合（MAG7 + CASH）
ASSET_TICKERS = MAG7_TICKERS + [CASH_TICKER]

# 实验配置
EXPERIMENT_CONFIG = {
    # 基本设置
    "initial_capital": 1_000_000,  # 本金 1M
    "simulation_days": 30,  # 模拟周期
    "tickers": ASSET_TICKERS,  # 包含 CASH
    
    # LLM 配置
    "debug_llm": os.getenv("LM_STUDIO_DEBUG_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
    "production_llm": os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-32b"),
    "llm_provider": "lmstudio",
    "lmstudio_base_url": os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
    "lmstudio_api_key": os.getenv("LM_STUDIO_API_KEY"),
    
    # TSFM 配置
    "tsfm_model": "amazon/chronos-2",
    "tsfm_lookback_days": 365,  # 过去一年股价
    "tsfm_prediction_horizons": [1, 5, 10, 15, 20],  # 1天, 1周, 2周, 3周, 4周
    "tsfm_quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
    
    # 输出格式配置
    "tsfm_output_formats": {
        "format_1": "numeric_30d",           # 数字，接下来30天
        "format_2": "ratio_30d",             # 比例，接下来30天
        "format_3": "ratio_multi_horizon",   # 比例，1天/1周/2周/3周/4周
        "format_4": "numeric_quantile_30d",  # 数字，分位数，30天
        "format_5": "ratio_quantile_30d",    # 比例，分位数，30天
        "format_6": "ratio_quantile_multi",  # 比例，分位数，多时间窗口
    },
    
    # 数据源
    "data_vendor": "alpha_vantage",
    
    # 结果保存
    "results_dir": "./experiment_results",
    "save_intermediate": True,
}

# 实验类型
class ExperimentType:
    BASELINE_LLM_ONLY = "baseline_llm_only"
    LLM_TSFM_FORMAT_1 = "llm_tsfm_format_1"
    LLM_TSFM_FORMAT_2 = "llm_tsfm_format_2"
    LLM_TSFM_FORMAT_3 = "llm_tsfm_format_3"
    LLM_TSFM_FORMAT_4 = "llm_tsfm_format_4"
    LLM_TSFM_FORMAT_5 = "llm_tsfm_format_5"
    LLM_TSFM_FORMAT_6 = "llm_tsfm_format_6"


def get_experiment_dir(experiment_type: str, run_id: str = None) -> str:
    """获取实验结果目录"""
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = EXPERIMENT_CONFIG["results_dir"]
    return os.path.join(base_dir, experiment_type, run_id)
