#!/usr/bin/env python3
"""
批量运行 BCSR 持续学习实验矩阵

实验配置：2方法 x 3数据集 x 3随机种子 = 18组实验

使用方式:
    # 串行运行（安全）
    python run_experiments.py --parallel 1

    # 并行运行（推荐，根据GPU数量调整）
    python run_experiments.py --parallel 2

    # 只运行特定方法
    python run_experiments.py --methods bcsr --datasets cifar-bs-50

    # 自定义配置
    python run_experiments.py --seeds 0 1 2 3 4 --parallel 3
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# 实验配置矩阵
METHODS = ['bcsr', 'uniform']  # coreset 方法需要 NTK，未实现
DATASETS = ['cifar-bs-50', 'imb-cifar-bs-50', 'noise-cifar-bs-50']
DEFAULT_SEEDS = [0, 1, 2]

# 超参数配置（对应论文Table 1-3）
HYPERPARAMS = {
    'cifar-bs-50': {
        'lr_decay': 0.9,
        'ref_hyp': 0.5,
        'lr_proxy_model': 5.0,
        'lr_weight': 5.0,
    },
    'imb-cifar-bs-50': {
        'lr_decay': 0.875,
        'ref_hyp': 0.1,
        'lr_proxy_model': 10.0,
        'lr_weight': 10.0,
    },
    'noise-cifar-bs-50': {
        'lr_decay': 0.875,
        'ref_hyp': 0.1,
        'lr_proxy_model': 10.0,
        'lr_weight': 10.0,
    },
}


def create_directories():
    """创建必要的目录"""
    dirs = ['data', 'summary', 'checkpoints', 'data/loss_data', 'logs']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print(f"✓ 目录已创建: {', '.join(dirs)}")


def run_command(cmd, log_file=None):
    """执行命令并输出到日志"""
    print(f"\n{'='*60}")
    print(f"执行命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    if log_file:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_errors='replace',
                text=True
            )

            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line, end='')
                    f.write(line)

            process.wait()
            return process.returncode
    else:
        return subprocess.call(cmd)


def build_exp_args(select_type, dataset, seed):
    """构建实验命令参数"""
    params = HYPERPARAMS[dataset]
    return [
        sys.executable, 'main.py',
        '--select_type', select_type,
        '--dataset', dataset,
        '--seed', str(seed),
        '--lr_decay', str(params['lr_decay']),
        '--ref_hyp', str(params['ref_hyp']),
        '--lr_proxy_model', str(params['lr_proxy_model']),
        '--lr_weight', str(params['lr_weight']),
    ]


def run_experiments(methods, datasets, seeds, parallel=1):
    """运行所有实验"""
    total = len(methods) * len(datasets) * len(seeds)
    print(f"\n实验配置：{len(methods)}方法 x {len(datasets)}数据集 x {len(seeds)}种子 = {total}组实验")
    print(f"并行度: {parallel}\n")

    # 创建实验列表
    experiments = []
    exp_id = 1
    for method in methods:
        for dataset in datasets:
            for seed in seeds:
                experiments.append({
                    'id': exp_id,
                    'method': method,
                    'dataset': dataset,
                    'seed': seed,
                    'args': build_exp_args(method, dataset, seed),
                })
                exp_id += 1

    # 按数据集分组（避免同时运行不同数据集的实验）
    grouped = {}
    for exp in experiments:
        ds = exp['dataset']
        if ds not in grouped:
            grouped[ds] = []
        grouped[ds].append(exp)

    # 运行实验
    results = []
    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"# 数据集: {dataset}")
        print(f"{'#'*60}\n")

        exps = grouped[dataset]
        for i in range(0, len(exps), parallel):
            batch = exps[i:i+parallel]
            procs = []

            for exp in batch:
                log_file = f"logs/{exp['dataset']}-{exp['method']}-seed{exp['seed']}.log"
                print(f"[{exp['id']}/{total}] {exp['method']} @ {exp['dataset']} (seed={exp['seed']}) -> {log_file}")

                proc = subprocess.Popen(
                    exp['args'],
                    stdout=open(log_file, 'w', encoding='utf-8'),
                    stderr=subprocess.STDOUT,
                )
                procs.append((exp, proc))

            # 等待当前批次完成
            for exp, proc in procs:
                returncode = proc.wait()
                results.append({
                    **exp,
                    'success': returncode == 0,
                    'returncode': returncode,
                })

                if returncode == 0:
                    print(f"✓ [{exp['id']}] 完成")
                else:
                    print(f"✗ [{exp['id']}] 失败 (code={returncode})")

    return results


def print_summary(results):
    """打印实验汇总"""
    print(f"\n{'='*60}")
    print("实验完成汇总")
    print(f"{'='*60}\n")

    total = len(results)
    success = sum(1 for r in results if r['success'])
    failed = total - success

    print(f"总计: {total} | 成功: {success} | 失败: {failed}\n")

    if failed > 0:
        print("失败的实验：")
        for r in results:
            if not r['success']:
                print(f"  - {r['method']} @ {r['dataset']} (seed={r['seed']})")
        print()

    # 按方法分组统计
    print("按方法统计：")
    for method in METHODS:
        method_results = [r for r in results if r['method'] == method]
        method_success = sum(1 for r in method_results if r['success'])
        print(f"  {method}: {method_success}/{len(method_results)} 成功")

    # 按数据集分组统计
    print("\n按数据集统计：")
    for dataset in DATASETS:
        dataset_results = [r for r in results if r['dataset'] == dataset]
        dataset_success = sum(1 for r in dataset_results if r['success'])
        print(f"  {dataset}: {dataset_success}/{len(dataset_results)} 成功")

    # 保存结果
    summary_file = f"logs/summary-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total': total,
            'success': success,
            'failed': failed,
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='批量运行 BCSR 持续学习实验')
    parser.add_argument('--methods', nargs='+', choices=METHODS + ['all'],
                        default=['all'], help='选择运行的方法 [bcsr, uniform, all]')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS + ['all'],
                        default=['all'], help='选择运行的数据集')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help='随机种子列表')
    parser.add_argument('--parallel', type=int, default=1,
                        help='并行运行数量（建议=GPU数量）')

    args = parser.parse_args()

    # 处理 'all' 选项
    methods = METHODS if 'all' in args.methods else args.methods
    datasets = DATASETS if 'all' in args.datasets else args.datasets

    print("="*60)
    print(" BCSR 持续学习批量实验脚本")
    print("="*60)
    print(f"方法: {', '.join(methods)}")
    print(f"数据集: {', '.join(datasets)}")
    print(f"种子: {args.seeds}")
    print(f"并行度: {args.parallel}")
    print("="*60)

    # 创建目录
    create_directories()

    # 运行实验
    results = run_experiments(methods, datasets, args.seeds, args.parallel)

    # 打印汇总
    print_summary(results)

    # 检查是否所有实验都成功
    if all(r['success'] for r in results):
        print("\n✓ 所有实验成功完成！")
        return 0
    else:
        print(f"\n✗ {sum(1 for r in results if not r['success'])} 个实验失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())
