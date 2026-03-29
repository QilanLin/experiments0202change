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

    def __call__(
        self,
        date: str,
        state: PortfolioState,
        *,
        asof_date: str | None = None,
    ) -> PortfolioDecision:
        print(f"[STAGE] Starting LLM decision for {date}", flush=True)

        context_date = asof_date or date
        context = self.market_context_provider.build(context_date, state)
        prepared_request = self.portfolio_agent.prepare_request(
            current_date=date,
            asof_date=context_date,
            fundamentals=context.fundamentals,
            price_history=context.price_history,
            tsfm_forecasts=context.tsfm_forecasts,
            current_weights=context.current_weights,
        )
        llm_input_payload = {
            "decision_date": date,
            "market_context_asof_date": context_date,
            "market_context": context.to_dict(),
            "messages": prepared_request["messages"],
            "prompt": prepared_request["prompt"],
            "input_token_count": prepared_request.get("input_token_count"),
            "input_token_count_source": prepared_request.get("input_token_count_source"),
            "input_token_budget": prepared_request.get("input_token_budget"),
            "input_token_over_budget": prepared_request.get("input_token_over_budget", False),
            "input_token_truncated": prepared_request.get("input_token_truncated", False),
            "input_token_truncation_strategy": prepared_request.get("input_token_truncation_strategy"),
            "experiment_type": self.portfolio_agent.experiment_type,
            "tsfm_format": self.portfolio_agent.tsfm_format,
        }
        llm_input_path = self.artifact_store.save_llm_input(
            llm_input_payload,
            decision_date=date,
        )
        print(f"[STAGE] Saved LLM input for {date} -> {llm_input_path}", flush=True)

        decision = self.portfolio_agent.decide_from_request(prepared_request)

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
