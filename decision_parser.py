from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from .config import MAG7_TICKERS, CASH_TICKER, ASSET_TICKERS
from .portfolio_models import PortfolioDecision


class DecisionParser:
    """负责解析 LLM 输出并生成 PortfolioDecision。"""

    def __init__(self, experiment_type: str, tsfm_format: Optional[int] = None):
        self.experiment_type = experiment_type
        self.tsfm_format = tsfm_format
        self.debug = False

    def parse(self, output: str, current_date: str, prompt: str = "") -> PortfolioDecision:
        """解析 LLM 输出。"""
        raw_output = output

        # 尝试提取JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试直接解析
            json_match = re.search(r'\{[^{}]*"weights"[^{}]*\{[^{}]*\}[^{}]*\}', output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return self._fallback_decision(current_date, raw_output, "Failed to parse JSON", prompt)

        try:
            data = json.loads(json_str)
            llm_weights_raw = data.get("weights", {})

            # 关键：在归一化前判断 LLM 是否显式提供了 CASH
            cash_provided = CASH_TICKER in llm_weights_raw

            if not cash_provided:
                # 路径1：LLM 没有显式给 CASH
                # 先对 MAG7 权重做 clip 到 [0,1]（缺失补 0）
                mag7_weights = {}
                for ticker in MAG7_TICKERS:
                    raw_val = llm_weights_raw.get(ticker, 0.0)
                    mag7_weights[ticker] = max(0.0, min(1.0, float(raw_val)))

                mag7_sum = sum(mag7_weights.values())

                if mag7_sum == 0:
                    # Fallback: MAG7 等权 + CASH=0
                    weights = {t: 1 / 7 for t in MAG7_TICKERS}
                    weights[CASH_TICKER] = 0.0
                elif mag7_sum <= 1.0:
                    # 直接设置 CASH = 1 - mag7_sum，保持 MAG7 原比例（不归一化）
                    weights = mag7_weights.copy()
                    weights[CASH_TICKER] = 1.0 - mag7_sum
                else:
                    # MAG7 总和 > 1，归一化 MAG7 到 1，CASH=0
                    weights = {t: mag7_weights[t] / mag7_sum for t in MAG7_TICKERS}
                    weights[CASH_TICKER] = 0.0
            else:
                # 路径2：LLM 显式给了 CASH
                # 对 8 个资产权重都 clip 到 [0,1]（缺失补 0）
                weights = {}
                for ticker in ASSET_TICKERS:
                    raw_val = llm_weights_raw.get(ticker, 0.0)
                    weights[ticker] = max(0.0, min(1.0, float(raw_val)))

                total = sum(weights.values())

                if total == 0:
                    # Fallback: MAG7 等权 + CASH=0
                    weights = {t: 1 / 7 for t in MAG7_TICKERS}
                    weights[CASH_TICKER] = 0.0
                else:
                    # 将所有 8 个资产一起归一化到和为 1（包含 CASH）
                    weights = {k: v / total for k, v in weights.items()}

            # 确保所有 8 个资产都有权重（防御性检查）
            for ticker in ASSET_TICKERS:
                if ticker not in weights:
                    weights[ticker] = 0.0

            # 最终检查：处理浮点误差（如果偏离 1 超过 1e-6，重新归一化）
            final_sum = sum(weights.values())
            if abs(final_sum - 1.0) > 1e-6:
                weights = {k: v / final_sum for k, v in weights.items()}

            if self.debug:
                print(
                    f"[DEBUG] DecisionParser: cash_provided={cash_provided}, "
                    f"mag7_sum={sum(weights.get(t, 0) for t in MAG7_TICKERS):.6f}, "
                    f"cash={weights.get(CASH_TICKER, 0):.6f}, final_sum={sum(weights.values()):.6f}"
                )

            return PortfolioDecision(
                decision_date=current_date,
                weights=weights,
                action=data.get("action", "rebalance"),
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", 0.5),
                raw_llm_output=raw_output,
                prompt=prompt,
                experiment_type=self.experiment_type,
                tsfm_format=self.tsfm_format,
            )
        except json.JSONDecodeError as e:
            return self._fallback_decision(current_date, raw_output, f"JSON decode error: {e}", prompt)

    def _fallback_decision(self, current_date: str, raw_output: str, error: str, prompt: str = "") -> PortfolioDecision:
        """Fallback: MAG7 等权重分配 + CASH=0。"""
        weights = {t: 1 / 7 for t in MAG7_TICKERS}
        weights[CASH_TICKER] = 0.0
        return PortfolioDecision(
            decision_date=current_date,
            weights=weights,
            action="hold",
            reasoning=f"Fallback due to: {error}",
            confidence=0.0,
            raw_llm_output=raw_output,
            prompt=prompt,
            experiment_type=self.experiment_type,
            tsfm_format=self.tsfm_format,
        )
