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

from .format_registry import TSFM_FORMAT_IDS, build_experiment_entries
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
        # 保持原行为：默认只跑 format_1 到 format_6
        tsfm_formats = [fmt for fmt in TSFM_FORMAT_IDS if fmt <= 6]
    
    results = []
    
    # 实验列表改由统一 registry 生成，避免这里再维护一份 format 映射
    experiments = build_experiment_entries(
        include_baseline=not skip_baseline,
        format_ids=tsfm_formats,
    )
    
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
        help="TSFM format IDs to test (defaults to all registered TSFM formats)"
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
