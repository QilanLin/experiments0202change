"""
Portfolio Weight Allocation Agent

LLM决策输出：MAG7股票的权重分配
"""

from __future__ import annotations
from typing import Dict, Any, Optional

from .config import MAG7_TICKERS
from .decision_parser import DecisionParser
from .portfolio_models import PortfolioDecision, PortfolioState
from .prompt_builder import PromptBuilder


class PortfolioWeightAgent:
    """组合权重决策Agent"""
    
    SYSTEM_PROMPT = """You are a portfolio manager for MAG7 stocks (AAPL, GOOGL, AMZN, MSFT, META, TSLA, NVDA) plus CASH. You should trade daily.

Your task is to allocate portfolio weights based on the provided information.

RULES:
1. Output weights for all 8 assets (7 stocks + CASH) that sum to exactly 1.0
2. Weights must be between 0.0 and 1.0
3. You can set weight to 0 if you want to avoid a stock
4. CASH represents the uninvested portion (cash held in account)
5. Consider risk diversification
6. You can hold cash (CASH > 0) if you want to reduce exposure

OUTPUT FORMAT (MUST follow exactly):
```json
{
  "action": "rebalance" or "hold",
  "weights": {
    "AAPL": 0.XX,
    "GOOGL": 0.XX,
    "AMZN": 0.XX,
    "MSFT": 0.XX,
    "META": 0.XX,
    "TSLA": 0.XX,
    "NVDA": 0.XX,
    "CASH": 0.XX
  },
  "confidence": 0.X,
  "reasoning": "Brief explanation"
}
```

IMPORTANT: Weights MUST sum to 1.0 exactly. All 8 assets (7 stocks + CASH) must be included."""

    def __init__(self, llm, experiment_type: str, tsfm_format: Optional[int] = None):
        self.llm = llm
        self.experiment_type = experiment_type
        self.tsfm_format = tsfm_format
        self.prompt_builder = PromptBuilder()
        self.decision_parser = DecisionParser(
            experiment_type=experiment_type,
            tsfm_format=tsfm_format,
        )

    def prepare_request(
            self,
            current_date: str,
            fundamentals: Dict[str, str],
            price_history: Dict[str, Any],
            tsfm_forecasts: Optional[Dict[str, str]] = None,
            current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """构造给 LLM 的结构化输入。"""
        user_context = self.prompt_builder.build_context(
            current_date, fundamentals, price_history,
            tsfm_forecasts, current_weights
        )
        messages = self.prompt_builder.build_messages(self.SYSTEM_PROMPT, user_context)
        prompt_str = self.prompt_builder.format_prompt_for_saving(messages)
        return {
            "decision_date": current_date,
            "messages": messages,
            "prompt": prompt_str,
        }

    def decide_from_request(self, prepared_request: Dict[str, Any]) -> PortfolioDecision:
        """使用已准备好的 messages 调 LLM，并解析输出。"""
        current_date = prepared_request["decision_date"]
        messages = prepared_request["messages"]
        prompt_str = prepared_request["prompt"]

        result = self.llm.invoke(messages)
        output = result.content if hasattr(result, 'content') else str(result)

        self.decision_parser.debug = getattr(self, "debug", False)
        return self.decision_parser.parse(output, current_date, prompt_str)

    def decide(
            self,
            current_date: str,
            fundamentals: Dict[str, str],
            price_history: Dict[str, Any],
            tsfm_forecasts: Optional[Dict[str, str]] = None,
            current_weights: Optional[Dict[str, float]] = None,
    ) -> PortfolioDecision:
        """做出权重决策"""
        prepared_request = self.prepare_request(
            current_date, fundamentals, price_history,
            tsfm_forecasts, current_weights
        )
        return self.decide_from_request(prepared_request)
