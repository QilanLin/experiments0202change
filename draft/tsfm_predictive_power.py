"""
TSFM 预测力诊断实验

评估 TSFM (Chronos-2) 对 MAG7 日线 close 的预测能力：
- 方向命中率 (direction hit rate)
- 相关性 (correlation)
- 校准覆盖率 (coverage)
- MAE / RMSE

使用方法：
    python -m experiments.tsfm_predictive_power --end-date 2024-01-15 --days 90 --horizons 5 20
    python -m experiments.tsfm_predictive_power --tsfm-outputs-dir ./experiment_results/llm_tsfm_format_1/20240115_143022/tsfm_outputs
"""

from __future__ import annotations
import os
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

from .data_loader import AlphaVantageLoader
from .tsfm_forecaster import TSFMForecaster, TSFMForecast
from .config import MAG7_TICKERS, get_experiment_dir


@dataclass
class EvalRow:
    """单条评估样本"""
    ticker: str
    date: str
    horizon: int
    p_t: float
    p_true: float
    p_hat: Optional[float] = None
    r_true: Optional[float] = None
    r_hat: Optional[float] = None
    hit: Optional[int] = None
    q05: Optional[float] = None
    q95: Optional[float] = None
    cover: Optional[int] = None
    status: str = "pending"
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


def _load_price_df(
    loader: AlphaVantageLoader,
    ticker: str,
    start_date: str,
    end_date: str,
    lookback_days: int
) -> pd.DataFrame:
    """加载价格数据"""
    df = loader.get_daily_prices(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days,
        use_cache=True
    )
    
    # 确保有 date 和 close 列
    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})
    if 'close' not in df.columns and 'Close' in df.columns:
        df = df.rename(columns={'Close': 'close'})
    
    # 按日期升序排列
    df = df.sort_values('date').reset_index(drop=True)
    
    # 确保 date 是字符串格式
    if pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    return df


def _iter_eval_points(
    df: pd.DataFrame,
    horizons: List[int],
    min_history: int
) -> Iterator[Tuple[int, str, pd.Series, float, Dict[int, float]]]:
    """
    迭代评估点
    
    Yields:
        (i, date_str, history_series, p_t, {horizon: p_future})
    """
    for i in range(len(df)):
        # 检查是否有足够的历史数据
        if i < min_history - 1:
            continue
        
        # 检查是否有足够的未来数据
        max_horizon = max(horizons)
        if i + max_horizon >= len(df):
            break
        
        date_str = str(df.iloc[i]['date'])
        p_t = float(df.iloc[i]['close'])
        
        # 提取历史序列（从 max(0, i-min_history+1) 到 i+1）
        start_idx = max(0, i - min_history + 1)
        history_series = df.iloc[start_idx:i+1]['close'].reset_index(drop=True)
        
        # 提取未来价格
        future_prices = {}
        for h in horizons:
            future_idx = i + h
            if future_idx < len(df):
                future_prices[h] = float(df.iloc[future_idx]['close'])
        
        yield (i, date_str, history_series, p_t, future_prices)


def _load_forecast_from_json(
    tsfm_outputs_dir: str,
    ticker: str,
    date_str: str
) -> Optional[TSFMForecast]:
    """从 JSON 文件加载预测结果"""
    filepath = os.path.join(tsfm_outputs_dir, f"{ticker}_{date_str}.json")
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # 从 dict 重建 TSFMForecast
        forecast = TSFMForecast(**data)
        return forecast
    except Exception as e:
        return None


def _get_forecast_for_date(
    ticker: str,
    date_str: str,
    history_series: pd.Series,
    forecaster: Optional[TSFMForecaster],
    tsfm_outputs_dir: Optional[str] = None
) -> Optional[TSFMForecast]:
    """获取指定日期的预测"""
    # 如果提供了 tsfm_outputs_dir，只从文件读取（不进行在线预测）
    if tsfm_outputs_dir:
        return _load_forecast_from_json(tsfm_outputs_dir, ticker, date_str)
    
    # 在线预测
    if forecaster is None:
        return None
    
    try:
        forecast = forecaster.forecast_all_formats(
            prices=history_series,
            ticker=ticker,
            forecast_date=date_str
        )
        return forecast
    except Exception as e:
        return None


def _extract_pred(
    forecast: TSFMForecast,
    h: int
) -> Tuple[Optional[float], Optional[float], Optional[float], str, Optional[str]]:
    """
    从预测结果中提取指定 horizon 的预测值
    
    Returns:
        (p_hat, q05, q95, status, error)
    """
    if forecast.status == "error":
        return None, None, None, "error", forecast.error
    
    try:
        # 检查索引是否有效 (h-1 因为索引从0开始)
        idx = h - 1
        if idx < 0 or idx >= 30:
            return None, None, None, "error", f"Invalid horizon index: {idx}"
        
        # 提取中位数预测价格
        if forecast.numeric_30d is None or len(forecast.numeric_30d) <= idx:
            return None, None, None, "error", "Missing numeric_30d"
        p_hat = float(forecast.numeric_30d[idx])
        
        # 提取分位数
        q05 = None
        q95 = None
        if forecast.numeric_quantile_30d is not None:
            q05_list = forecast.numeric_quantile_30d.get("0.05")
            q95_list = forecast.numeric_quantile_30d.get("0.95")
            if q05_list is not None and len(q05_list) > idx:
                q05 = float(q05_list[idx])
            if q95_list is not None and len(q95_list) > idx:
                q95 = float(q95_list[idx])
        
        status = forecast.status
        error = None
        
        return p_hat, q05, q95, status, error
        
    except Exception as e:
        return None, None, None, "error", str(e)


def compute_metrics(per_point_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """计算聚合指标"""
    # 只使用有效样本（status != "error"）
    valid_df = per_point_df[per_point_df['status'] != 'error'].copy()
    
    if len(valid_df) == 0:
        # 返回空的结果
        summary_cols = ['ticker', 'horizon', 'n', 'hit_rate', 'corr', 'coverage_90', 'mae', 'rmse']
        summary_df = pd.DataFrame(columns=summary_cols)
        overall_dict = {
            'n': 0,
            'hit_rate': None,
            'corr': None,
            'coverage_90': None,
            'mae': None,
            'rmse': None
        }
        return summary_df, overall_dict
    
    # 按 (ticker, horizon) 聚合
    summary_rows = []
    
    for (ticker, horizon), group in valid_df.groupby(['ticker', 'horizon']):
        n = len(group)
        
        # 方向命中率
        hit_col = group['hit']
        hit_rate = hit_col.mean() if hit_col.notna().any() else None
        
        # 相关性（只使用同时有效的样本）
        corr_data = group[['r_true', 'r_hat']].dropna()
        if len(corr_data) >= 2:
            corr = np.corrcoef(corr_data['r_true'], corr_data['r_hat'])[0, 1]
        else:
            corr = None
        
        # 覆盖率
        cover_col = group['cover']
        coverage_90 = cover_col.mean() if cover_col.notna().any() else None
        
        # MAE / RMSE（只使用同时有效的样本）
        mae_data = group[['p_true', 'p_hat']].dropna()
        if len(mae_data) > 0:
            errors = mae_data['p_true'] - mae_data['p_hat']
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors ** 2))
        else:
            mae = None
            rmse = None
        
        summary_rows.append({
            'ticker': ticker,
            'horizon': horizon,
            'n': n,
            'hit_rate': hit_rate,
            'corr': corr,
            'coverage_90': coverage_90,
            'mae': mae,
            'rmse': rmse
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # 整体汇总
    overall_dict = {}
    if len(valid_df) > 0:
        overall_dict['n'] = len(valid_df)
        
        hit_col = valid_df['hit']
        overall_dict['hit_rate'] = hit_col.mean() if hit_col.notna().any() else None
        
        corr_data = valid_df[['r_true', 'r_hat']].dropna()
        if len(corr_data) >= 2:
            overall_dict['corr'] = np.corrcoef(corr_data['r_true'], corr_data['r_hat'])[0, 1]
        else:
            overall_dict['corr'] = None
        
        cover_col = valid_df['cover']
        overall_dict['coverage_90'] = cover_col.mean() if cover_col.notna().any() else None
        
        mae_data = valid_df[['p_true', 'p_hat']].dropna()
        if len(mae_data) > 0:
            errors = mae_data['p_true'] - mae_data['p_hat']
            overall_dict['mae'] = np.mean(np.abs(errors))
            overall_dict['rmse'] = np.sqrt(np.mean(errors ** 2))
        else:
            overall_dict['mae'] = None
            overall_dict['rmse'] = None
    
    return summary_df, overall_dict


def main():
    parser = argparse.ArgumentParser(description="TSFM 预测力诊断实验")
    parser.add_argument("--tickers", nargs="+", default=None, help="股票代码列表，默认使用 MAG7_TICKERS")
    parser.add_argument("--end-date", type=str, default=None, help="结束日期 (YYYY-MM-DD)，默认今天")
    parser.add_argument("--days", type=int, default=90, help="评估天数，默认90")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20], help="预测horizon列表，默认 [5, 20]")
    parser.add_argument("--min-history", type=int, default=40, help="最小历史长度，默认40")
    parser.add_argument("--use-mock", action="store_true", help="使用mock预测器")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    parser.add_argument("--tsfm-outputs-dir", type=str, default=None, help="TSFM输出目录（优先读取）")
    parser.add_argument("--max-points-per-ticker", type=int, default=None, help="每个ticker最大评估点数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（用于采样）")
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # 解析参数
    tickers = args.tickers or MAG7_TICKERS
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    horizons = args.horizons
    min_history = args.min_history
    use_mock = args.use_mock
    device = args.device
    tsfm_outputs_dir = args.tsfm_outputs_dir
    max_points_per_ticker = args.max_points_per_ticker
    
    # 计算开始日期
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=args.days + min_history)  # 多留一些历史数据
    start_date = start_dt.strftime("%Y-%m-%d")
    
    # 初始化加载器和预测器
    loader = AlphaVantageLoader()
    forecaster = None
    if not tsfm_outputs_dir:
        forecaster = TSFMForecaster(device=device, use_mock=use_mock)
    
    # 创建输出目录
    experiment_type = "tsfm_predictive_power"
    results_dir = get_experiment_dir(experiment_type)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"结果将保存到: {results_dir}")
    print(f"评估配置: tickers={tickers}, end_date={end_date}, days={args.days}, horizons={horizons}")
    print(f"min_history={min_history}, use_mock={use_mock}, tsfm_outputs_dir={tsfm_outputs_dir}")
    
    # 收集所有评估点
    eval_rows = []
    
    for ticker in tickers:
        print(f"\n处理 {ticker}...")
        
        try:
            # 加载价格数据
            df = _load_price_df(loader, ticker, start_date, end_date, args.days + min_history)
            if len(df) == 0:
                print(f"  {ticker}: 无数据，跳过")
                continue
            
            print(f"  {ticker}: 加载了 {len(df)} 条价格记录")
            
            # 生成评估点
            eval_points = list(_iter_eval_points(df, horizons, min_history))
            
            # 采样（如果指定了最大点数）
            if max_points_per_ticker and len(eval_points) > max_points_per_ticker:
                indices = np.random.choice(len(eval_points), max_points_per_ticker, replace=False)
                eval_points = [eval_points[i] for i in sorted(indices)]
                print(f"  {ticker}: 采样到 {len(eval_points)} 个评估点")
            
            # 处理每个评估点
            for point_idx, (i, date_str, history_series, p_t, future_prices) in enumerate(eval_points):
                if (point_idx + 1) % 10 == 0:
                    print(f"  {ticker}: 处理进度 {point_idx + 1}/{len(eval_points)}")
                
                # 获取预测
                forecast = _get_forecast_for_date(
                    ticker, date_str, history_series, forecaster, tsfm_outputs_dir
                )
                
                if forecast is None:
                    # 创建错误记录
                    for h in horizons:
                        if h in future_prices:
                            eval_rows.append(EvalRow(
                                ticker=ticker,
                                date=date_str,
                                horizon=h,
                                p_t=p_t,
                                p_true=future_prices[h],
                                status="error",
                                error="Failed to get forecast"
                            ))
                    continue
                
                # 对每个 horizon 评估
                for h in horizons:
                    if h not in future_prices:
                        continue
                    
                    p_true = future_prices[h]
                    
                    # 提取预测值
                    p_hat, q05, q95, status, error = _extract_pred(forecast, h)
                    
                    # 计算指标
                    r_true = None
                    r_hat = None
                    hit = None
                    cover = None
                    
                    if p_hat is not None and forecast.last_close is not None:
                        # 计算收益率
                        r_true = (p_true - p_t) / p_t
                        r_hat = (p_hat - p_t) / p_t
                        
                        # 方向命中率
                        hit = 1 if (r_hat * r_true) > 0 else 0
                        
                        # 覆盖率
                        if q05 is not None and q95 is not None:
                            cover = 1 if (q05 <= p_true <= q95) else 0
                    
                    eval_rows.append(EvalRow(
                        ticker=ticker,
                        date=date_str,
                        horizon=h,
                        p_t=p_t,
                        p_true=p_true,
                        p_hat=p_hat,
                        r_true=r_true,
                        r_hat=r_hat,
                        hit=hit,
                        q05=q05,
                        q95=q95,
                        cover=cover,
                        status=status,
                        error=error
                    ))
            
            print(f"  {ticker}: 完成，共 {len([r for r in eval_rows if r.ticker == ticker])} 条记录")
            
        except Exception as e:
            print(f"  {ticker}: 错误 - {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 转换为 DataFrame
    print(f"\n处理完成，共 {len(eval_rows)} 条评估记录")
    
    if len(eval_rows) == 0:
        print("警告: 没有评估记录，退出")
        return
    
    per_point_df = pd.DataFrame([row.to_dict() for row in eval_rows])
    
    # 计算聚合指标
    print("计算聚合指标...")
    summary_df, overall_dict = compute_metrics(per_point_df)
    
    # 保存结果
    print("保存结果...")
    
    # per_point.csv
    per_point_path = os.path.join(results_dir, "per_point.csv")
    per_point_df.to_csv(per_point_path, index=False)
    print(f"  per_point.csv: {per_point_path}")
    
    # summary_by_ticker_horizon.csv
    summary_path = os.path.join(results_dir, "summary_by_ticker_horizon.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  summary_by_ticker_horizon.csv: {summary_path}")
    
    # summary_overall.json
    config_dict = {
        'end_date': end_date,
        'days': args.days,
        'horizons': horizons,
        'min_history': min_history,
        'use_mock': use_mock,
        'tsfm_outputs_dir': tsfm_outputs_dir,
        'tickers': tickers,
        'max_points_per_ticker': max_points_per_ticker,
        'seed': args.seed,
    }
    overall_dict_with_config = {
        'config': config_dict,
        'overall_metrics': overall_dict
    }
    overall_path = os.path.join(results_dir, "summary_overall.json")
    with open(overall_path, 'w') as f:
        json.dump(overall_dict_with_config, f, indent=2, default=str)
    print(f"  summary_overall.json: {overall_path}")
    
    # report.md
    report_path = os.path.join(results_dir, "report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# TSFM 预测力诊断实验报告\n\n")
        f.write(f"**运行时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 配置\n\n")
        f.write(f"- 结束日期: {end_date}\n")
        f.write(f"- 评估天数: {args.days}\n")
        f.write(f"- Horizons: {horizons}\n")
        f.write(f"- 最小历史长度: {min_history}\n")
        f.write(f"- 使用 Mock: {use_mock}\n")
        f.write(f"- TSFM输出目录: {tsfm_outputs_dir or '在线预测'}\n")
        f.write(f"- Tickers: {', '.join(tickers)}\n\n")
        
        f.write("## 整体指标\n\n")
        if overall_dict['n'] > 0:
            f.write(f"- 总样本数: {overall_dict['n']}\n")
            f.write(f"- 方向命中率: {overall_dict['hit_rate']:.4f if overall_dict['hit_rate'] is not None else 'N/A'}\n")
            f.write(f"- 相关性: {overall_dict['corr']:.4f if overall_dict['corr'] is not None else 'N/A'}\n")
            f.write(f"- 覆盖率 (90%): {overall_dict['coverage_90']:.4f if overall_dict['coverage_90'] is not None else 'N/A'}\n")
            f.write(f"- MAE: {overall_dict['mae']:.4f if overall_dict['mae'] is not None else 'N/A'}\n")
            f.write(f"- RMSE: {overall_dict['rmse']:.4f if overall_dict['rmse'] is not None else 'N/A'}\n\n")
        else:
            f.write("无有效样本\n\n")
        
        f.write("## 按 Ticker 和 Horizon 汇总\n\n")
        if len(summary_df) > 0:
            # 手动生成 Markdown 表格
            cols = summary_df.columns.tolist()
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
            for _, row in summary_df.iterrows():
                values = []
                for col in cols:
                    val = row[col]
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        values.append("N/A")
                    elif isinstance(val, float):
                        values.append(f"{val:.4f}")
                    else:
                        values.append(str(val))
                f.write("| " + " | ".join(values) + " |\n")
            f.write("\n")
        else:
            f.write("无汇总数据\n\n")
        
        f.write("## 详细数据\n\n")
        f.write(f"- per_point.csv: 每条样本的详细记录\n")
        f.write(f"- summary_by_ticker_horizon.csv: 按 (ticker, horizon) 聚合的指标\n")
        f.write(f"- summary_overall.json: 整体指标和配置\n")
    
    print(f"  report.md: {report_path}")
    print(f"\n实验完成！结果保存在: {results_dir}")


if __name__ == "__main__":
    main()
