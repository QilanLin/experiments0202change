#!/usr/bin/env python
"""
完整实验运行脚本 - GPU服务器专用

用法:
    python experiments/run_full_experiment.py
    python experiments/run_full_experiment.py --production  # 使用30B模型
    python experiments/run_full_experiment.py --days 60     # 60天模拟
"""

from __future__ import annotations
import argparse
import os
import sys
import json
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    
    # 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("✗ GPU不可用，将使用CPU（速度会很慢）")
    except ImportError:
        print("✗ PyTorch未安装")
        return False
    
    # 检查API Key
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if api_key:
        print(f"✓ Alpha Vantage API Key已设置")
    else:
        print("✗ ALPHA_VANTAGE_API_KEY未设置")
        return False
    
    # 检查Chronos
    try:
        from chronos import Chronos2Pipeline
        print("✓ Chronos-2已安装")
    except ImportError:
        print("⚠ Chronos-2未安装，TSFM将使用fallback模式")
    
    # 检查Qwen
    try:
        from tradingagents.llms.local_qwen import LocalQwenChat
        print("✓ LocalQwenChat可用")
    except ImportError:
        print("✗ LocalQwenChat不可用")
        return False
    
    print("=" * 60)
    return True


def run_experiments(
    use_production_model: bool = False,
    simulation_days: int = 30,
    end_date: str = None,
):
    """运行完整实验"""
    from experiments.config import EXPERIMENT_CONFIG
    from experiments.format_registry import build_experiment_entries
    from experiments.run_experiment import ExperimentRunner
    from experiments.compare_results import find_all_results, generate_report
    
    # 实验列表改由统一 registry 生成，避免脚本内手写一份 baseline / format 定义
    experiments = build_experiment_entries(format_ids=[1, 2, 3, 4, 5, 6])
    
    # 选择模型
    debug_mode = not use_production_model
    model_name = EXPERIMENT_CONFIG["debug_llm"] if debug_mode else EXPERIMENT_CONFIG["production_llm"]
    
    print("\n" + "=" * 60)
    print("实验配置")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"模拟天数: {simulation_days}")
    print(f"结束日期: {end_date or '今天'}")
    print(f"实验数量: {len(experiments)}")
    print("=" * 60 + "\n")
    
    results = []
    start_time = time.time()
    
    for i, exp in enumerate(experiments, 1):
        exp_start = time.time()
        print(f"\n{'='*60}")
        print(f"[{i}/{len(experiments)}] 运行实验: {exp['name']}")
        print(f"{'='*60}")
        
        try:
            runner = ExperimentRunner(
                experiment_type=exp["type"],
                debug=debug_mode,
                simulation_days=simulation_days,
                tsfm_format=exp["tsfm_format"],
                use_mock_llm=False,  # 使用真实LLM
            )
            
            result = runner.run(end_date=end_date)
            
            exp_time = time.time() - exp_start
            results.append({
                "experiment": exp["name"],
                "type": exp["type"],
                "tsfm_format": exp["tsfm_format"],
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "max_drawdown": result.max_drawdown,
                "status": "success",
                "time_seconds": exp_time,
            })
            
            print(f"\n✓ 完成: {exp['name']}")
            print(f"  收益率: {result.total_return*100:.2f}%")
            print(f"  Sharpe: {result.sharpe_ratio:.3f}")
            print(f"  用时: {exp_time/60:.1f}分钟")
            
        except Exception as e:
            exp_time = time.time() - exp_start
            print(f"\n✗ 失败: {exp['name']}")
            print(f"  错误: {str(e)}")
            results.append({
                "experiment": exp["name"],
                "type": exp["type"],
                "tsfm_format": exp["tsfm_format"],
                "status": "failed",
                "error": str(e),
                "time_seconds": exp_time,
            })
    
    total_time = time.time() - start_time
    
    # 保存汇总结果
    summary_path = os.path.join(
        "./experiment_results",
        f"full_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    summary = {
        "run_time": datetime.now().isoformat(),
        "total_time_minutes": total_time / 60,
        "config": {
            "model": model_name,
            "simulation_days": simulation_days,
            "end_date": end_date,
            "use_production_model": use_production_model,
        },
        "results": results,
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    # 打印最终汇总
    print("\n" + "=" * 60)
    print("实验完成汇总")
    print("=" * 60)
    print(f"总用时: {total_time/60:.1f}分钟")
    print(f"成功: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
    print(f"结果保存: {summary_path}")
    
    # 打印结果表格
    print("\n性能对比:")
    print("-" * 60)
    print(f"{'实验':<35} {'收益率':>10} {'Sharpe':>10}")
    print("-" * 60)
    for r in results:
        if r['status'] == 'success':
            print(f"{r['experiment']:<35} {r['total_return']*100:>9.2f}% {r['sharpe_ratio']:>10.3f}")
        else:
            print(f"{r['experiment']:<35} {'FAILED':>10} {'-':>10}")
    print("-" * 60)
    
    # 生成对比报告
    print("\n生成对比报告...")
    all_results = find_all_results("./experiment_results")
    if all_results:
        report_path = os.path.join("./experiment_results", "comparison_report.txt")
        generate_report(all_results, report_path)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="运行完整的LLM vs LLM+TSFM实验")
    parser.add_argument(
        "--production", 
        action="store_true", 
        help="使用生产模型(30B)，否则使用debug模型(4B)"
    )
    parser.add_argument(
        "--days", 
        type=int, 
        default=30, 
        help="模拟天数 (默认30)"
    )
    parser.add_argument(
        "--end-date", 
        type=str, 
        default=None, 
        help="结束日期 (YYYY-MM-DD格式，默认今天)"
    )
    parser.add_argument(
        "--skip-check", 
        action="store_true", 
        help="跳过环境检查"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("LLM vs LLM+TSFM 完整实验")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 环境检查
    if not args.skip_check:
        if not check_environment():
            print("\n环境检查失败，请修复上述问题后重试")
            print("或使用 --skip-check 跳过检查")
            sys.exit(1)
    
    # 运行实验
    try:
        results = run_experiments(
            use_production_model=args.production,
            simulation_days=args.days,
            end_date=args.end_date,
        )
        
        # 检查是否有失败
        failed = [r for r in results if r['status'] == 'failed']
        if failed:
            print(f"\n⚠ 有{len(failed)}个实验失败")
            sys.exit(1)
        else:
            print("\n✓ 所有实验成功完成!")
            
    except KeyboardInterrupt:
        print("\n\n实验被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n实验出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
