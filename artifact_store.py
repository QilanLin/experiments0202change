from __future__ import annotations

import json
import os
from typing import Any, Optional


class ArtifactStore:
    """统一管理实验运行期产物的保存路径与落盘行为。"""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def tsfm_input_dir(self, input_subdir: Optional[str] = None) -> str:
        """返回 TSFM 输入目录；历史可靠性回放会挂到其子目录下。"""
        base_dir = os.path.join(self.results_dir, "tsfm_inputs")
        return os.path.join(base_dir, input_subdir) if input_subdir else base_dir

    def save_tsfm_input(
        self,
        context_dict: dict[str, Any],
        *,
        ticker: str,
        forecast_date: str,
        input_subdir: Optional[str] = None,
    ) -> str:
        """保存喂给 TSFM 的输入快照，保持原有文件名和目录结构不变。"""
        target_dir = self.tsfm_input_dir(input_subdir)
        os.makedirs(target_dir, exist_ok=True)
        input_filename = os.path.join(target_dir, f"tsfm_input_{ticker}_{forecast_date}.json")
        with open(input_filename, "w", encoding="utf-8") as f:
            json.dump(context_dict, f, indent=2, default=str)
        return input_filename

    def save_tsfm_output(self, forecast: Any, *, ticker: str, forecast_date: str) -> str:
        """保存每个 ticker / 交易日对应的 TSFM 输出。"""
        output_path = os.path.join(
            self.results_dir, "tsfm_outputs", f"{ticker}_{forecast_date}.json"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(forecast.to_dict(), f, indent=2, default=str)
        return output_path

    def save_llm_decision(self, decision: Any, *, decision_date: str) -> str:
        """保存每天的 LLM 决策输出（含 prompt 和 raw output）。"""
        output_path = os.path.join(
            self.results_dir, "llm_outputs", f"decision_{decision_date}.json"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(decision.to_dict(), f, indent=2, default=str)
        return output_path

    def simulation_result_path(self) -> str:
        """返回 simulation_result.json 的标准路径。"""
        return os.path.join(self.results_dir, "simulation_result.json")

    def save_simulation_result(self, result: Any) -> str:
        """保存最终回测结果。"""
        result_path = self.simulation_result_path()
        result.save(result_path)
        return result_path
