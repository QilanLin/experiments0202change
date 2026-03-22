from __future__ import annotations

from typing import Any, Dict

import requests


class AlphaVantageClient:
    """Alpha Vantage HTTP 客户端。"""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not set")
        self.api_key = api_key

    def make_request(self, params: Dict[str, str]) -> Any:
        """发送 Alpha Vantage 请求并做基础错误检查。"""
        req_params = dict(params)
        req_params["apikey"] = self.api_key
        response = requests.get(self.BASE_URL, params=req_params)
        response.raise_for_status()

        if req_params.get("datatype") == "csv":
            return response.text

        data = response.json()
        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        if "Information" in data and "premium" in data["Information"].lower():
            raise ValueError(f"API Error: {data['Information']}")
        return data
