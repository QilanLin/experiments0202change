"""
Data Loader - 数据加载模块

使用 Alpha Vantage API 获取：
- 股价数据（过去一年）
- 公司基本面数据
"""

from __future__ import annotations
import os
from typing import Dict, Any, List
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from .alpha_vantage_client import AlphaVantageClient
from .config import MAG7_TICKERS
from .data_repositories import FundamentalsRepository, PriceRepository


class AlphaVantageLoader:
    """Alpha Vantage 数据加载器"""

    def __init__(self, api_key: str = None, cache_dir: str = "./data_cache"):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not set")

        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "price"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "fundamentals"), exist_ok=True)
        self.client = AlphaVantageClient(self.api_key)
        self.price_repository = PriceRepository(self.client, cache_dir)
        self.fundamentals_repository = FundamentalsRepository(self.client, cache_dir)

    def get_daily_prices(
        self, 
        ticker: str, 
        start_date: str = None,
        end_date: str = None,
        lookback_days: int = 365,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        return self.price_repository.get_daily_prices(
            ticker,
            start_date=start_date,
            end_date=end_date,
            lookback_days=lookback_days,
            use_cache=use_cache,
        )
    
    def get_fundamentals(self, ticker: str, use_cache: bool = True) -> Dict[str, Any]:
        return self.fundamentals_repository.get_fundamentals(ticker, use_cache=use_cache)
    
    def get_income_statement(self, ticker: str, use_cache: bool = True) -> Dict[str, Any]:
        return self.fundamentals_repository.get_income_statement(ticker, use_cache=use_cache)
    
    def get_balance_sheet(self, ticker: str, use_cache: bool = True) -> Dict[str, Any]:
        return self.fundamentals_repository.get_balance_sheet(ticker, use_cache=use_cache)
    
    def load_all_data(
        self,
        tickers: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        lookback_days: int = 365,
    ) -> Dict[str, Any]:
        """加载所有数据"""
        if tickers is None:
            tickers = MAG7_TICKERS
        
        result = {
            "prices": {},
            "fundamentals": {},
        }
        
        for ticker in tickers:
            # 跳过 CASH，不需要拉取价格数据
            if ticker == "CASH":
                continue
                
            print(f"Loading data for {ticker}...")
            
            try:
                result["prices"][ticker] = self.get_daily_prices(
                    ticker, start_date, end_date, lookback_days
                )
            except Exception as e:
                print(f"  Error loading prices for {ticker}: {e}")
            
            try:
                result["fundamentals"][ticker] = self.get_fundamentals(ticker)
            except Exception as e:
                print(f"  Error loading fundamentals for {ticker}: {e}")
        
        return result
    
    def format_fundamentals_for_llm(self, fundamentals: Dict[str, Any]) -> str:
        return self.fundamentals_repository.format_fundamentals_for_llm(fundamentals)
    
    def get_simple_fundamentals_asof(
        self, 
        ticker: str, 
        asof_date: str, 
        lag_days: int = 45,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        return self.fundamentals_repository.get_simple_fundamentals_asof(
            ticker=ticker,
            asof_date=asof_date,
            lag_days=lag_days,
            use_cache=use_cache,
        )
    
    def format_simple_fundamentals_for_llm(self, snapshot: Dict[str, Any]) -> str:
        return self.fundamentals_repository.format_simple_fundamentals_for_llm(snapshot)
