"""
Data Loader - 数据加载模块

使用 Alpha Vantage API 获取：
- 股价数据（过去一年）
- 公司基本面数据
"""

from __future__ import annotations
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import requests
from io import StringIO

from dotenv import load_dotenv
load_dotenv()

from .config import MAG7_TICKERS


class AlphaVantageLoader:
    """Alpha Vantage 数据加载器"""
    
    BASE_URL = "https://www.alphavantage.co/query"
    RATE_LIMIT_DELAY = 1  # 减少延迟，依赖缓存
    
    def __init__(self, api_key: str = None, cache_dir: str = "./data_cache"):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not set")
        
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "price"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "fundamentals"), exist_ok=True)
    
    def _make_request(self, params: Dict[str, str]) -> Any:
        """发送API请求"""
        params["apikey"] = self.api_key
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        
        # 检查是否是CSV
        if params.get("datatype") == "csv":
            return response.text
        
        # JSON响应 - 检查错误
        data = response.json()
        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        if "Information" in data and "premium" in data["Information"].lower():
            raise ValueError(f"API Error: {data['Information']}")
        
        return data
    
    def _get_cache_path(self, data_type: str, ticker: str, suffix: str = "") -> str:
        """获取缓存文件路径"""
        filename = f"{ticker}{suffix}.json" if data_type != "price" else f"{ticker}{suffix}.csv"
        return os.path.join(self.cache_dir, data_type, filename)
    
    def _load_from_cache(self, cache_path: str, allow_stale: bool = False) -> Optional[Any]:
        """从缓存加载"""
        if os.path.exists(cache_path):
            # 检查缓存是否过期（1天）
            mtime = os.path.getmtime(cache_path)
            if allow_stale or time.time() - mtime < 86400:
                if cache_path.endswith(".csv"):
                    return pd.read_csv(cache_path)
                with open(cache_path, 'r') as f:
                    return json.load(f)
        return None
    
    def _save_to_cache(self, cache_path: str, data: Any):
        """保存到缓存"""
        if isinstance(data, pd.DataFrame):
            data.to_csv(cache_path, index=False)
        else:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def get_daily_prices(
        self, 
        ticker: str, 
        start_date: str = None,
        end_date: str = None,
        lookback_days: int = 365,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """获取日线数据"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=lookback_days)
            start_date = start_dt.strftime("%Y-%m-%d")
        
        # 免费版 Alpha Vantage 仅支持 TIME_SERIES_DAILY + compact（最近约100个交易日）。
        # 使用 plain_daily 后缀区分当前缓存格式，避免与旧的 premium adjusted 缓存混淆。
        cache_path = self._get_cache_path("price", ticker, f"_plain_daily_{start_date}_{end_date}")
        
        if use_cache:
            cached = self._load_from_cache(cache_path)
            if cached is not None:
                return cached
        
        # API请求 - 使用免费可用的日线接口。
        # 注意：TIME_SERIES_DAILY_ADJUSTED 和 outputsize=full 在当前免费 key 下都会触发 premium 限制。
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "compact",  # 免费版：最近约100个交易日
            "datatype": "csv",
        }
        
        try:
            csv_data = self._make_request(params)
        except Exception:
            stale_cached = self._load_from_cache(cache_path, allow_stale=True) if use_cache else None
            if stale_cached is not None:
                print(f"  Falling back to stale price cache for {ticker}")
                return stale_cached
            raise
        
        # 检查是否返回了错误信息（JSON格式）
        if csv_data.strip().startswith('{'):
            import json
            error_data = json.loads(csv_data)
            error_msg = error_data.get("Information", error_data.get("Error Message", "Unknown error"))
            raise ValueError(f"API Error: {error_msg}")
        
        df = pd.read_csv(StringIO(csv_data))
        
        # 免费接口只返回原始 close；为保持下游兼容，统一保证存在 close 列。
        # 如果未来切换回其它接口并带有 adjusted_close，则优先使用 adjusted_close。
        if 'adjusted_close' in df.columns:
            # 保留原始 close 到 raw_close（用于debug）
            if 'close' in df.columns:
                df['raw_close'] = df['close'].copy()
            # 将 adjusted_close 赋值给 close（保证下游代码兼容）
            df['close'] = df['adjusted_close'].copy()
        elif 'adjusted close' in df.columns:
            # 处理列名带空格的情况
            if 'close' in df.columns:
                df['raw_close'] = df['close'].copy()
            df['close'] = df['adjusted close'].copy()
            df = df.drop(columns=['adjusted close'])
        elif 'close' in df.columns:
            df['raw_close'] = df['close'].copy()
        
        # 过滤日期范围
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            date_col = 'timestamp'
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            date_col = 'date'
        else:
            raise ValueError(f"No date column found in data. Columns: {df.columns.tolist()}")
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # 统一列名为 'date'
        if date_col == 'timestamp':
            df = df.rename(columns={'timestamp': 'date'})
        
        # 确保至少包含 'date' 和 'close' 列
        if 'close' not in df.columns:
            raise ValueError(f"No 'close' column found after processing. Columns: {df.columns.tolist()}")
        
        self._save_to_cache(cache_path, df)
        time.sleep(self.RATE_LIMIT_DELAY)
        
        return df
    
    def get_fundamentals(self, ticker: str, use_cache: bool = True) -> Dict[str, Any]:
        """获取公司基本面数据"""
        cache_path = self._get_cache_path("fundamentals", ticker, "_overview")
        
        if use_cache:
            cached = self._load_from_cache(cache_path)
            if cached is not None:
                return cached
        
        params = {
            "function": "OVERVIEW",
            "symbol": ticker,
        }
        
        try:
            data = self._make_request(params)
        except Exception:
            stale_cached = self._load_from_cache(cache_path, allow_stale=True) if use_cache else None
            if stale_cached is not None:
                print(f"  Falling back to stale fundamentals cache for {ticker} overview")
                return stale_cached
            raise
        self._save_to_cache(cache_path, data)
        time.sleep(self.RATE_LIMIT_DELAY)
        
        return data
    
    def get_income_statement(self, ticker: str, use_cache: bool = True) -> Dict[str, Any]:
        """获取利润表"""
        cache_path = self._get_cache_path("fundamentals", ticker, "_income")
        
        if use_cache:
            cached = self._load_from_cache(cache_path)
            if cached is not None:
                return cached
        
        params = {
            "function": "INCOME_STATEMENT",
            "symbol": ticker,
        }
        
        try:
            data = self._make_request(params)
        except Exception:
            stale_cached = self._load_from_cache(cache_path, allow_stale=True) if use_cache else None
            if stale_cached is not None:
                print(f"  Falling back to stale fundamentals cache for {ticker} income")
                return stale_cached
            raise
        self._save_to_cache(cache_path, data)
        time.sleep(self.RATE_LIMIT_DELAY)
        
        return data
    
    def get_balance_sheet(self, ticker: str, use_cache: bool = True) -> Dict[str, Any]:
        """获取资产负债表"""
        cache_path = self._get_cache_path("fundamentals", ticker, "_balance")
        
        if use_cache:
            cached = self._load_from_cache(cache_path)
            if cached is not None:
                return cached
        
        params = {
            "function": "BALANCE_SHEET",
            "symbol": ticker,
        }
        
        try:
            data = self._make_request(params)
        except Exception:
            stale_cached = self._load_from_cache(cache_path, allow_stale=True) if use_cache else None
            if stale_cached is not None:
                print(f"  Falling back to stale fundamentals cache for {ticker} balance")
                return stale_cached
            raise
        self._save_to_cache(cache_path, data)
        time.sleep(self.RATE_LIMIT_DELAY)
        
        return data
    
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
        """格式化基本面数据供LLM使用（保留用于向后兼容）"""
        if not fundamentals:
            return "No fundamental data available"
        
        key_metrics = [
            ("Name", "Name"),
            ("Sector", "Sector"),
            ("Industry", "Industry"),
            ("MarketCap", "MarketCapitalization"),
            ("PE Ratio", "PERatio"),
            ("PEG Ratio", "PEGRatio"),
            ("Book Value", "BookValue"),
            ("Dividend Yield", "DividendYield"),
            ("EPS", "EPS"),
            ("Revenue TTM", "RevenueTTM"),
            ("Profit Margin", "ProfitMargin"),
            ("Operating Margin", "OperatingMarginTTM"),
            ("ROE", "ReturnOnEquityTTM"),
            ("ROA", "ReturnOnAssetsTTM"),
            ("52 Week High", "52WeekHigh"),
            ("52 Week Low", "52WeekLow"),
            ("50 Day MA", "50DayMovingAverage"),
            ("200 Day MA", "200DayMovingAverage"),
            ("Beta", "Beta"),
        ]
        
        lines = []
        for label, key in key_metrics:
            value = fundamentals.get(key, "N/A")
            if value and value != "None":
                lines.append(f"{label}: {value}")
        
        return "\n".join(lines)
    
    def _parse_float(self, value: Any) -> Optional[float]:
        """安全解析浮点数（处理 Alpha Vantage 返回的字符串数值，可能为 'None'）"""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            if value.strip() in ("None", "", "N/A", "n/a"):
                return None
            try:
                # 移除可能的逗号分隔符
                cleaned = value.replace(",", "").strip()
                return float(cleaned)
            except (ValueError, AttributeError):
                return None
        return None
    
    def get_simple_fundamentals_asof(
        self, 
        ticker: str, 
        asof_date: str, 
        lag_days: int = 45,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        获取截至指定日期的可用最近一期基本面数据（as-of 基本面）
        
        注意：Alpha Vantage 的 OVERVIEW 接口没有 date 参数，返回的是"最新快照"，
        对历史回测会引入未来信息。因此必须使用财报序列（INCOME_STATEMENT / BALANCE_SHEET）
        在本地做 as-of 选择。这些财报接口只有 function/symbol/apikey，没有 date 参数。
        LISTING_STATUS 才有 date，但那不是基本面数据。
        
        参数:
            ticker: 股票代码
            asof_date: 截止日期（字符串，格式：YYYY-MM-DD）
            lag_days: 财报发布延迟天数（默认45天，表示财报在 fiscalDateEnding + lag_days 后才可用）
            use_cache: 是否使用缓存
        
        返回:
            简化的基本面字典，包含截至 asof_date 可用的最近一期财报数据
        """
        asof_dt = pd.to_datetime(asof_date)
        cutoff_date = asof_dt - pd.Timedelta(days=lag_days)
        
        # 获取利润表和资产负债表（使用缓存）
        income_statement = self.get_income_statement(ticker, use_cache=use_cache)
        balance_sheet = self.get_balance_sheet(ticker, use_cache=use_cache)
        
        # 从季度报告中筛选"可用的最近一期"
        # 条件：fiscalDateEnding <= asof_date - lag_days
        eligible_income = []
        if "quarterlyReports" in income_statement:
            for report in income_statement["quarterlyReports"]:
                fiscal_date_str = report.get("fiscalDateEnding")
                if fiscal_date_str:
                    try:
                        fiscal_date = pd.to_datetime(fiscal_date_str)
                        if fiscal_date <= cutoff_date:
                            eligible_income.append((fiscal_date, report))
                    except (ValueError, TypeError):
                        continue
        
        eligible_balance = []
        if "quarterlyReports" in balance_sheet:
            for report in balance_sheet["quarterlyReports"]:
                fiscal_date_str = report.get("fiscalDateEnding")
                if fiscal_date_str:
                    try:
                        fiscal_date = pd.to_datetime(fiscal_date_str)
                        if fiscal_date <= cutoff_date:
                            eligible_balance.append((fiscal_date, report))
                    except (ValueError, TypeError):
                        continue
        
        # 选择最新的可用报告（基于 fiscalDateEnding）
        income_report = None
        balance_report = None
        
        if eligible_income:
            _, income_report = max(eligible_income, key=lambda x: x[0])
        
        if eligible_balance:
            _, balance_report = max(eligible_balance, key=lambda x: x[0])
        
        # 如果没有找到合适的报告，尝试使用最近的报告（即使超过 cutoff_date）
        # 这是 graceful fallback
        if income_report is None and "quarterlyReports" in income_statement:
            if income_statement["quarterlyReports"]:
                income_report = income_statement["quarterlyReports"][0]
        
        if balance_report is None and "quarterlyReports" in balance_sheet:
            if balance_sheet["quarterlyReports"]:
                balance_report = balance_sheet["quarterlyReports"][0]
        
        # 构建简化的基本面字典
        snapshot = {}
        
        # 选择最近的 fiscalDateEnding（优先用 income，否则用 balance）
        fiscal_date_ending = None
        if income_report:
            fiscal_date_ending = income_report.get("fiscalDateEnding")
        elif balance_report:
            fiscal_date_ending = balance_report.get("fiscalDateEnding")
        snapshot["fiscalDateEnding"] = fiscal_date_ending
        
        # 从利润表提取数据
        if income_report:
            snapshot["totalRevenue"] = self._parse_float(income_report.get("totalRevenue"))
            snapshot["grossProfit"] = self._parse_float(income_report.get("grossProfit"))
            snapshot["netIncome"] = self._parse_float(income_report.get("netIncome"))
        else:
            snapshot["totalRevenue"] = None
            snapshot["grossProfit"] = None
            snapshot["netIncome"] = None
        
        # 从资产负债表提取数据
        if balance_report:
            snapshot["totalAssets"] = self._parse_float(balance_report.get("totalAssets"))
            snapshot["totalLiabilities"] = self._parse_float(balance_report.get("totalLiabilities"))
        else:
            snapshot["totalAssets"] = None
            snapshot["totalLiabilities"] = None
        
        # 计算比率
        if snapshot.get("totalRevenue") and snapshot.get("totalRevenue") > 0:
            if snapshot.get("grossProfit") is not None:
                snapshot["grossMargin"] = snapshot["grossProfit"] / snapshot["totalRevenue"]
            else:
                snapshot["grossMargin"] = None
            
            if snapshot.get("netIncome") is not None:
                snapshot["netMargin"] = snapshot["netIncome"] / snapshot["totalRevenue"]
            else:
                snapshot["netMargin"] = None
        else:
            snapshot["grossMargin"] = None
            snapshot["netMargin"] = None
        
        if snapshot.get("totalAssets") and snapshot.get("totalAssets") > 0:
            if snapshot.get("totalLiabilities") is not None:
                snapshot["leverage"] = snapshot["totalLiabilities"] / snapshot["totalAssets"]
            else:
                snapshot["leverage"] = None
        else:
            snapshot["leverage"] = None
        
        return snapshot
    
    def format_simple_fundamentals_for_llm(self, snapshot: Dict[str, Any]) -> str:
        """
        格式化简化基本面数据供LLM使用（只包含 as-of 字段，不含未来信息）
        
        参数:
            snapshot: get_simple_fundamentals_asof() 返回的字典
        
        返回:
            格式化的多行文本字符串
        """
        if not snapshot:
            return "No fundamental data available"
        
        lines = []
        
        # 财报日期
        if snapshot.get("fiscalDateEnding"):
            lines.append(f"Fiscal Date Ending: {snapshot['fiscalDateEnding']}")
        
        # 收入相关
        if snapshot.get("totalRevenue") is not None:
            lines.append(f"Total Revenue: ${snapshot['totalRevenue']:,.0f}")
        
        if snapshot.get("grossProfit") is not None:
            lines.append(f"Gross Profit: ${snapshot['grossProfit']:,.0f}")
        
        if snapshot.get("netIncome") is not None:
            lines.append(f"Net Income: ${snapshot['netIncome']:,.0f}")
        
        # 比率
        if snapshot.get("grossMargin") is not None:
            lines.append(f"Gross Margin: {snapshot['grossMargin']*100:.2f}%")
        
        if snapshot.get("netMargin") is not None:
            lines.append(f"Net Margin: {snapshot['netMargin']*100:.2f}%")
        
        # 资产负债表相关
        if snapshot.get("totalAssets") is not None:
            lines.append(f"Total Assets: ${snapshot['totalAssets']:,.0f}")
        
        if snapshot.get("totalLiabilities") is not None:
            lines.append(f"Total Liabilities: ${snapshot['totalLiabilities']:,.0f}")
        
        if snapshot.get("leverage") is not None:
            lines.append(f"Leverage (Liabilities/Assets): {snapshot['leverage']*100:.2f}%")
        
        if not lines:
            return "No fundamental data available"
        
        return "\n".join(lines)
