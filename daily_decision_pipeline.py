from __future__ import annotations

from .artifact_store import ArtifactStore
from .market_context import MarketContextProvider
from .portfolio_agent import PortfolioWeightAgent
from .portfolio_models import PortfolioDecision, PortfolioState


class DailyDecisionPipeline:
    """封装单个交易日的上下文准备、LLM 决策和产物落盘。"""

    def __init__(
        self,
        *,
        market_context_provider: MarketContextProvider,
        portfolio_agent: PortfolioWeightAgent,
        artifact_store: ArtifactStore,
        debug: bool = False,
    ):
        self.market_context_provider = market_context_provider
        self.portfolio_agent = portfolio_agent
        self.artifact_store = artifact_store
        self.debug = debug

    def __call__(self, date: str, state: PortfolioState) -> PortfolioDecision:
        print(f"[STAGE] Starting LLM decision for {date}", flush=True)

        context = self.market_context_provider.build(date, state)

        decision = self.portfolio_agent.decide(
            current_date=date,
            fundamentals=context.fundamentals,
            price_history=context.price_history,
            tsfm_forecasts=context.tsfm_forecasts,
            current_weights=context.current_weights,
        )

        if self.debug:
            weights_sum = sum(decision.weights.values())
            print(f"[DEBUG] Date: {date}")
            print(f"  Parsed weights: {decision.weights}")
            print(f"  Weights sum: {weights_sum:.6f}")
            if abs(weights_sum - 1.0) > 0.01:
                print("  WARNING: Weights sum is not 1.0!")

        llm_output_path = self.artifact_store.save_llm_decision(
            decision,
            decision_date=date,
        )
        print(f"[STAGE] Saved LLM decision for {date} -> {llm_output_path}", flush=True)

        return decision
