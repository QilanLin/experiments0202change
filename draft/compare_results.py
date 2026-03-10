"""
Compare Results - 实验结果对比分析

用法:
    python -m experiments.compare_results --results-dir ./experiment_results
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd

from .config import ExperimentType


def load_result(result_path: str) -> Dict[str, Any]:
    """加载单个实验结果"""
    with open(result_path, 'r') as f:
        return json.load(f)


def find_all_results(results_dir: str) -> List[Dict[str, Any]]:
    """查找所有实验结果"""
    results = []
    
    for exp_type in os.listdir(results_dir):
        exp_dir = os.path.join(results_dir, exp_type)
        if not os.path.isdir(exp_dir):
            continue
        
        for run_id in os.listdir(exp_dir):
            run_dir = os.path.join(exp_dir, run_id)
            result_file = os.path.join(run_dir, "simulation_result.json")
            
            if os.path.exists(result_file):
                result = load_result(result_file)
                result["run_id"] = run_id
                result["result_path"] = result_file
                results.append(result)
    
    return results


def compare_experiments(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """对比实验结果"""
    rows = []
    
    for r in results:
        rows.append({
            "Experiment": r["experiment_type"],
            "Run ID": r.get("run_id", "N/A"),
            "Period": f"{r['start_date']} to {r['end_date']}",
            "Initial Capital": r["initial_capital"],
            "Final Value": r["final_value"],
            "Total Return (%)": r["total_return"] * 100,
            "Annualized Return (%)": r["annualized_return"] * 100,
            "Sharpe Ratio": r["sharpe_ratio"],
            "Sortino Ratio": r["sortino_ratio"],
            "Max Drawdown (%)": r["max_drawdown"] * 100,
            "Num Trades": r["num_trades"],
            "Num Decisions": r["num_decisions"],
        })
    
    df = pd.DataFrame(rows)
    return df.sort_values("Total Return (%)", ascending=False)


def generate_report(results: List[Dict[str, Any]], output_path: str = None):
    """生成对比报告"""
    df = compare_experiments(results)
    
    report = []
    report.append("=" * 80)
    report.append("EXPERIMENT COMPARISON REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    
    # 摘要表格
    report.append("PERFORMANCE SUMMARY")
    report.append("-" * 80)
    
    summary_cols = [
        "Experiment", "Total Return (%)", "Sharpe Ratio", 
        "Sortino Ratio", "Max Drawdown (%)"
    ]
    summary_df = df[summary_cols].copy()
    summary_df = summary_df.round(2)
    report.append(summary_df.to_string(index=False))
    report.append("")
    
    # 最佳表现
    report.append("BEST PERFORMERS")
    report.append("-" * 80)
    
    best_return = df.loc[df["Total Return (%)"].idxmax()]
    report.append(f"Highest Return: {best_return['Experiment']} ({best_return['Total Return (%)']:.2f}%)")
    
    best_sharpe = df.loc[df["Sharpe Ratio"].idxmax()]
    report.append(f"Best Sharpe: {best_sharpe['Experiment']} ({best_sharpe['Sharpe Ratio']:.3f})")
    
    best_sortino = df.loc[df["Sortino Ratio"].idxmax()]
    report.append(f"Best Sortino: {best_sortino['Experiment']} ({best_sortino['Sortino Ratio']:.3f})")
    
    lowest_dd = df.loc[df["Max Drawdown (%)"].idxmin()]
    report.append(f"Lowest Drawdown: {lowest_dd['Experiment']} ({lowest_dd['Max Drawdown (%)']:.2f}%)")
    report.append("")
    
    # Baseline对比
    baseline_results = df[df["Experiment"].str.contains("baseline", case=False)]
    tsfm_results = df[~df["Experiment"].str.contains("baseline", case=False)]
    
    if len(baseline_results) > 0 and len(tsfm_results) > 0:
        report.append("TSFM vs BASELINE COMPARISON")
        report.append("-" * 80)
        
        baseline_return = baseline_results["Total Return (%)"].mean()
        tsfm_return = tsfm_results["Total Return (%)"].mean()
        
        report.append(f"Baseline Avg Return: {baseline_return:.2f}%")
        report.append(f"TSFM Avg Return: {tsfm_return:.2f}%")
        report.append(f"TSFM Improvement: {tsfm_return - baseline_return:.2f}%")
        report.append("")
        
        # 按TSFM格式分组
        report.append("BY TSFM FORMAT:")
        for _, row in tsfm_results.iterrows():
            diff = row["Total Return (%)"] - baseline_return
            sign = "+" if diff > 0 else ""
            report.append(f"  {row['Experiment']}: {row['Total Return (%)']:.2f}% ({sign}{diff:.2f}% vs baseline)")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {output_path}")
    
    print(report_text)
    return df


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default="./experiment_results",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output path for comparison report"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        return
    
    results = find_all_results(args.results_dir)
    
    if len(results) == 0:
        print("No experiment results found")
        return
    
    print(f"Found {len(results)} experiment results")
    
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(args.results_dir, "comparison_report.txt")
    
    generate_report(results, output_path)


if __name__ == "__main__":
    main()
