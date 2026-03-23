from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .alpha_vantage_client import AlphaVantageClient


class CachedRepository:
    """带本地缓存的仓储基类。"""

    def __init__(self, cache_dir: str, data_type: str):
        self.cache_dir = cache_dir
        self.data_type = data_type
        os.makedirs(os.path.join(cache_dir, data_type), exist_ok=True)

    def get_cache_path(self, ticker: str, suffix: str = "") -> str:
        ext = ".csv" if self.data_type == "price" else ".json"
        return os.path.join(self.cache_dir, self.data_type, f"{ticker}{suffix}{ext}")

    def load_from_cache(self, cache_path: str, allow_stale: bool = False) -> Optional[Any]:
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            if allow_stale or time.time() - mtime < 86400:
                if cache_path.endswith(".csv"):
                    return pd.read_csv(cache_path)
                with open(cache_path, "r") as f:
                    return json.load(f)
        return None

    def save_to_cache(self, cache_path: str, data: Any):
        if isinstance(data, pd.DataFrame):
            data.to_csv(cache_path, index=False)
        else:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)


class PriceRepository(CachedRepository):
    """价格数据仓储：本地缓存 + API 回退。"""

    RATE_LIMIT_DELAY = 1

    def __init__(self, client: AlphaVantageClient, cache_dir: str):
        super().__init__(cache_dir, "price")
        self.client = client

    def _normalize_price_cache_df(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None

        if "timestamp" in df.columns:
            date_col = "timestamp"
        elif "date" in df.columns:
            date_col = "date"
        elif "Date" in df.columns:
            date_col = "Date"
        else:
            return None

        normalized = df.copy()
        normalized[date_col] = pd.to_datetime(normalized[date_col], errors="coerce")
        normalized = normalized.dropna(subset=[date_col])
        if normalized.empty:
            return None

        if "adjusted_close" in normalized.columns:
            if "close" in normalized.columns:
                normalized["raw_close"] = normalized["close"].copy()
            normalized["close"] = normalized["adjusted_close"].copy()
        elif "adjusted close" in normalized.columns:
            if "close" in normalized.columns:
                normalized["raw_close"] = normalized["close"].copy()
            normalized["close"] = normalized["adjusted close"].copy()

        if "close" not in normalized.columns:
            return None

        normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce")
        normalized = normalized.dropna(subset=["close"])
        if normalized.empty:
            return None

        normalized = normalized.rename(columns={date_col: "date"})
        normalized = normalized.sort_values("date").reset_index(drop=True)
        return normalized

    def _load_exact_cache_snapshot(self, cache_path: str) -> Optional[pd.DataFrame]:
        """优先复用请求对应的精确缓存快照。

        对回测来说，历史价格文件一旦落盘，本身就是一个固定输入快照；这里即使
        mtime 已经过期，也优先复用 exact cache，而不是切换到另一份覆盖更广的
        本地 CSV。这样能最大限度保持旧实验的可复现性。
        """
        if not os.path.exists(cache_path):
            return None

        try:
            normalized = self._normalize_price_cache_df(pd.read_csv(cache_path))
        except Exception:
            return None
        return normalized

    def _load_best_local_price_cache(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        price_dir = Path(self.cache_dir) / "price"
        if not price_dir.exists():
            return None

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        candidates = []

        for cache_file in sorted(price_dir.glob(f"{ticker}*.csv")):
            try:
                normalized = self._normalize_price_cache_df(pd.read_csv(cache_file))
            except Exception:
                continue
            if normalized is None or normalized.empty:
                continue

            file_start = normalized["date"].min()
            file_end = normalized["date"].max()
            covers_range = file_start <= start_dt and file_end >= end_dt
            overlaps_range = file_end >= start_dt and file_start <= end_dt
            if not overlaps_range:
                continue

            sliced = normalized[
                (normalized["date"] >= start_dt) & (normalized["date"] <= end_dt)
            ].copy()
            if sliced.empty:
                continue

            candidates.append(
                {
                    "path": str(cache_file),
                    "df": sliced.reset_index(drop=True),
                    "covers_range": covers_range,
                    "rows": len(sliced),
                    "file_start": file_start,
                    "file_end": file_end,
                }
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item["covers_range"],
                item["rows"],
                item["file_end"],
                item["file_start"],
            ),
            reverse=True,
        )
        best = candidates[0]
        print(f"  Using local cached prices for {ticker} from {best['path']}")
        return best["df"]

    def get_daily_prices(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        lookback_days: int = 365,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=lookback_days)
            start_date = start_dt.strftime("%Y-%m-%d")

        cache_path = self.get_cache_path(ticker, f"_plain_daily_{start_date}_{end_date}")

        if use_cache:
            cached = self.load_from_cache(cache_path)
            if cached is not None:
                return cached
            exact_snapshot = self._load_exact_cache_snapshot(cache_path)
            if exact_snapshot is not None:
                print(f"  Using exact cached prices for {ticker} from {cache_path}")
                return exact_snapshot
            best_local_cache = self._load_best_local_price_cache(ticker, start_date, end_date)
            if best_local_cache is not None:
                return best_local_cache

        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "compact",
            "datatype": "csv",
        }

        try:
            csv_data = self.client.make_request(params)
        except Exception:
            stale_cached = self.load_from_cache(cache_path, allow_stale=True) if use_cache else None
            if stale_cached is not None:
                print(f"  Falling back to stale price cache for {ticker}")
                return stale_cached
            raise

        if csv_data.strip().startswith("{"):
            error_data = json.loads(csv_data)
            error_msg = error_data.get("Information", error_data.get("Error Message", "Unknown error"))
            raise ValueError(f"API Error: {error_msg}")

        df = pd.read_csv(StringIO(csv_data))
        if "adjusted_close" in df.columns:
            if "close" in df.columns:
                df["raw_close"] = df["close"].copy()
            df["close"] = df["adjusted_close"].copy()
        elif "adjusted close" in df.columns:
            if "close" in df.columns:
                df["raw_close"] = df["close"].copy()
            df["close"] = df["adjusted close"].copy()
            df = df.drop(columns=["adjusted close"])
        elif "close" in df.columns:
            df["raw_close"] = df["close"].copy()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            date_col = "timestamp"
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            date_col = "date"
        else:
            raise ValueError(f"No date column found in data. Columns: {df.columns.tolist()}")

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]
        df = df.sort_values(date_col).reset_index(drop=True)

        if date_col == "timestamp":
            df = df.rename(columns={"timestamp": "date"})
        if "close" not in df.columns:
            raise ValueError(f"No 'close' column found after processing. Columns: {df.columns.tolist()}")

        self.save_to_cache(cache_path, df)
        time.sleep(self.RATE_LIMIT_DELAY)
        return df


class FundamentalsRepository(CachedRepository):
    """基本面数据仓储：overview/财报/as-of 快照。"""

    RATE_LIMIT_DELAY = 1

    def __init__(self, client: AlphaVantageClient, cache_dir: str):
        super().__init__(cache_dir, "fundamentals")
        self.client = client

    def _get_json_document(
        self,
        *,
        ticker: str,
        suffix: str,
        function: str,
        fallback_label: str,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        cache_path = self.get_cache_path(ticker, suffix)
        if use_cache:
            cached = self.load_from_cache(cache_path)
            if cached is not None:
                return cached

        params = {"function": function, "symbol": ticker}
        try:
            data = self.client.make_request(params)
        except Exception:
            stale_cached = self.load_from_cache(cache_path, allow_stale=True) if use_cache else None
            if stale_cached is not None:
                print(f"  Falling back to stale fundamentals cache for {ticker} {fallback_label}")
                return stale_cached
            raise

        self.save_to_cache(cache_path, data)
        time.sleep(self.RATE_LIMIT_DELAY)
        return data

    def get_fundamentals(self, ticker: str, use_cache: bool = True) -> Dict[str, Any]:
        return self._get_json_document(
            ticker=ticker,
            suffix="_overview",
            function="OVERVIEW",
            fallback_label="overview",
            use_cache=use_cache,
        )

    def get_income_statement(self, ticker: str, use_cache: bool = True) -> Dict[str, Any]:
        return self._get_json_document(
            ticker=ticker,
            suffix="_income",
            function="INCOME_STATEMENT",
            fallback_label="income",
            use_cache=use_cache,
        )

    def get_balance_sheet(self, ticker: str, use_cache: bool = True) -> Dict[str, Any]:
        return self._get_json_document(
            ticker=ticker,
            suffix="_balance",
            function="BALANCE_SHEET",
            fallback_label="balance",
            use_cache=use_cache,
        )

    def format_fundamentals_for_llm(self, fundamentals: Dict[str, Any]) -> str:
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
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            if value.strip() in ("None", "", "N/A", "n/a"):
                return None
            try:
                return float(value.replace(",", "").strip())
            except (ValueError, AttributeError):
                return None
        return None

    def get_simple_fundamentals_asof(
        self,
        ticker: str,
        asof_date: str,
        lag_days: int = 45,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        asof_dt = pd.to_datetime(asof_date)
        cutoff_date = asof_dt - pd.Timedelta(days=lag_days)

        income_statement = self.get_income_statement(ticker, use_cache=use_cache)
        balance_sheet = self.get_balance_sheet(ticker, use_cache=use_cache)

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

        income_report = max(eligible_income, key=lambda x: x[0])[1] if eligible_income else None
        balance_report = max(eligible_balance, key=lambda x: x[0])[1] if eligible_balance else None

        if income_report is None and "quarterlyReports" in income_statement and income_statement["quarterlyReports"]:
            income_report = income_statement["quarterlyReports"][0]
        if balance_report is None and "quarterlyReports" in balance_sheet and balance_sheet["quarterlyReports"]:
            balance_report = balance_sheet["quarterlyReports"][0]

        snapshot: Dict[str, Any] = {}
        fiscal_date_ending = None
        if income_report:
            fiscal_date_ending = income_report.get("fiscalDateEnding")
        elif balance_report:
            fiscal_date_ending = balance_report.get("fiscalDateEnding")
        snapshot["fiscalDateEnding"] = fiscal_date_ending

        if income_report:
            snapshot["totalRevenue"] = self._parse_float(income_report.get("totalRevenue"))
            snapshot["grossProfit"] = self._parse_float(income_report.get("grossProfit"))
            snapshot["netIncome"] = self._parse_float(income_report.get("netIncome"))
        else:
            snapshot["totalRevenue"] = None
            snapshot["grossProfit"] = None
            snapshot["netIncome"] = None

        if balance_report:
            snapshot["totalAssets"] = self._parse_float(balance_report.get("totalAssets"))
            snapshot["totalLiabilities"] = self._parse_float(balance_report.get("totalLiabilities"))
        else:
            snapshot["totalAssets"] = None
            snapshot["totalLiabilities"] = None

        if snapshot.get("totalRevenue") and snapshot.get("totalRevenue") > 0:
            snapshot["grossMargin"] = (
                snapshot["grossProfit"] / snapshot["totalRevenue"]
                if snapshot.get("grossProfit") is not None
                else None
            )
            snapshot["netMargin"] = (
                snapshot["netIncome"] / snapshot["totalRevenue"]
                if snapshot.get("netIncome") is not None
                else None
            )
        else:
            snapshot["grossMargin"] = None
            snapshot["netMargin"] = None

        if snapshot.get("totalAssets") and snapshot.get("totalAssets") > 0:
            snapshot["leverage"] = (
                snapshot["totalLiabilities"] / snapshot["totalAssets"]
                if snapshot.get("totalLiabilities") is not None
                else None
            )
        else:
            snapshot["leverage"] = None

        return snapshot

    def format_simple_fundamentals_for_llm(self, snapshot: Dict[str, Any]) -> str:
        if not snapshot:
            return "No fundamental data available"

        lines = []
        if snapshot.get("fiscalDateEnding"):
            lines.append(f"Fiscal Date Ending: {snapshot['fiscalDateEnding']}")
        if snapshot.get("totalRevenue") is not None:
            lines.append(f"Total Revenue: ${snapshot['totalRevenue']:,.0f}")
        if snapshot.get("grossProfit") is not None:
            lines.append(f"Gross Profit: ${snapshot['grossProfit']:,.0f}")
        if snapshot.get("netIncome") is not None:
            lines.append(f"Net Income: ${snapshot['netIncome']:,.0f}")
        if snapshot.get("grossMargin") is not None:
            lines.append(f"Gross Margin: {snapshot['grossMargin']*100:.2f}%")
        if snapshot.get("netMargin") is not None:
            lines.append(f"Net Margin: {snapshot['netMargin']*100:.2f}%")
        if snapshot.get("totalAssets") is not None:
            lines.append(f"Total Assets: ${snapshot['totalAssets']:,.0f}")
        if snapshot.get("totalLiabilities") is not None:
            lines.append(f"Total Liabilities: ${snapshot['totalLiabilities']:,.0f}")
        if snapshot.get("leverage") is not None:
            lines.append(f"Leverage (Liabilities/Assets): {snapshot['leverage']*100:.2f}%")
        return "\n".join(lines) if lines else "No fundamental data available"
