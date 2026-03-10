"""
Run All Experiments - 批量运行所有实验

用法:
    python -m experiments.run_all_experiments --debug --days 30
"""

from __future__ import annotations
import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any

from .config import ExperimentType, get_experiment_dir
from .run_experiment import ExperimentRunner
from .compare_results import find_all_results, generate_report


def run_all_experiments(
    debug: bool = True,
    simulation_days: int = 30,
    end_date: str = None,
    skip_baseline: bool = False,
    tsfm_formats: List[int] = None,
    use_mock: bool = False,
) -> List[Dict[str, Any]]:
    """运行所有实验"""
    
    if tsfm_formats is None:
        tsfm_formats = [1, 2, 3, 4, 5, 6]
    
    results = []
    
    # 实验配置
    experiments = []
    
    if not skip_baseline:
        experiments.append({
            "type": ExperimentType.BASELINE_LLM_ONLY,
            "tsfm_format": None,
            "name": "Baseline (LLM Only)",
        })
    
    format_names = {
        1: "TSFM Format 1 (Numeric 30d)",
        2: "TSFM Format 2 (Ratio 30d)",
        3: "TSFM Format 3 (Ratio Multi-Horizon)",
        4: "TSFM Format 4 (Numeric Quantile 30d)",
        5: "TSFM Format 5 (Ratio Quantile 30d)",
        6: "TSFM Format 6 (Ratio Quantile Multi-Horizon)",
    }
    
    type_mapping = {
        1: ExperimentType.LLM_TSFM_FORMAT_1,
        2: ExperimentType.LLM_TSFM_FORMAT_2,
        3: ExperimentType.LLM_TSFM_FORMAT_3,
        4: ExperimentType.LLM_TSFM_FORMAT_4,
        5: ExperimentType.LLM_TSFM_FORMAT_5,
        6: ExperimentType.LLM_TSFM_FORMAT_6,
    }
    
    for fmt in tsfm_formats:
        experiments.append({
            "type": type_mapping[fmt],
            "tsfm_format": fmt,
            "name": format_names[fmt],
        })
    
    print(f"\n{'='*60}")
    print("BATCH EXPERIMENT RUNNER")
    print(f"{'='*60}")
    print(f"Total Experiments: {len(experiments)}")
    print(f"Debug Mode: {debug}")
    print(f"Mock Mode: {use_mock}")
    print(f"Simulation Days: {simulation_days}")
    print(f"End Date: {end_date or 'Today'}")
    print(f"{'='*60}\n")
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Running: {exp['name']}")
        print("-" * 40)
        
        try:
            runner = ExperimentRunner(
                experiment_type=exp["type"],
                debug=debug,
                simulation_days=simulation_days,
                tsfm_format=exp["tsfm_format"],
                use_mock_llm=use_mock,
            )
            
            result = runner.run(end_date=end_date)
            results.append({
                "experiment": exp["name"],
                "type": exp["type"],
                "tsfm_format": exp["tsfm_format"],
                "result": result.to_dict(),
                "status": "success",
            })
            
            print(f"✓ Completed: {exp['name']}")
            print(f"  Return: {result.total_return*100:.2f}%")
            print(f"  Sharpe: {result.sharpe_ratio:.3f}")
            
        except Exception as e:
            print(f"✗ Failed: {exp['name']}")
            print(f"  Error: {str(e)}")
            results.append({
                "experiment": exp["name"],
                "type": exp["type"],
                "tsfm_format": exp["tsfm_format"],
                "status": "failed",
                "error": str(e),
            })
    
    # 保存批量运行结果
    batch_result_path = os.path.join(
        "./experiment_results",
        f"batch_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(batch_result_path), exist_ok=True)
    
    with open(batch_result_path, 'w') as f:
        json.dump({
            "run_time": datetime.now().isoformat(),
            "config": {
                "debug": debug,
                "simulation_days": simulation_days,
                "end_date": end_date,
            },
            "results": results,
        }, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("BATCH RUN COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"Results saved to: {batch_result_path}")
    
    # 生成对比报告
    print("\nGenerating comparison report...")
    all_results = find_all_results("./experiment_results")
    if all_results:
        generate_report(all_results)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run all portfolio weight experiments")
    parser.add_argument("--debug", action="store_true", help="Use debug LLM")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM/TSFM for debugging")
    parser.add_argument("--days", type=int, default=30, help="Simulation days")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline experiment")
    parser.add_argument(
        "--formats", 
        type=int, 
        nargs="+", 
        default=None,
        help="TSFM formats to test (1-6)"
    )
    
    args = parser.parse_args()
    
    run_all_experiments(
        debug=args.debug,
        simulation_days=args.days,
        end_date=args.end_date,
        skip_baseline=args.skip_baseline,
        tsfm_formats=args.formats,
        use_mock=args.mock,
    )


if __name__ == "__main__":
    main()
