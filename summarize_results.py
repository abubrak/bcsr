#!/usr/bin/env python3
"""
汇总实验结果

从 summary/ 目录收集所有实验的 JSON 结果文件，生成汇总表格。

使用方式:
    python summarize_results.py
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import sys


def parse_exp_name(filename):
    """从文件名解析实验配置"""
    # 文件名格式: YYYY-MM-DD-HH-MM-SS-{dataset}-{method}_seqlr..._seqep..._seqbs...
    match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})-(cifar-bs-50|imb-cifar-bs-50|noise-cifar-bs-50|cifar10-bs-50|imb-cifar10-bs-50|noise-cifar10-bs-50|mnist-bs-50|imb-mnist-bs-50|noise-mnist-bs-50)-(bcsr|herding|gradmatch|gss|coreset|uniform)_', filename)
    if match:
        return {
            'timestamp': match.group(1),
            'dataset': match.group(2),
            'method': match.group(3),
        }
    return None


def load_results(summary_dir='summary'):
    """加载所有实验结果"""
    summary_path = Path(summary_dir)
    if not summary_path.exists():
        print(f"错误: 目录不存在 {summary_dir}")
        return []

    results = []
    for file in summary_path.glob('*'):
        if file.is_file():
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 从文件名解析配置
                config = parse_exp_name(file.name)
                if config:
                    results.append({
                        'file': file.name,
                        'method': config['method'],
                        'dataset': config['dataset'],
                        'avg_acc': data.get('AVG ACC', 0),
                        'forgetting': data.get('Forgetting', 0),
                        'per_task_accuracy': data.get('per-task accuracy', []),
                        'acc_avg': data.get('acc_avg', []),
                        'time': data.get('time', 0),
                    })
            except Exception as e:
                print(f"警告: 无法解析 {file.name}: {e}")

    return results


def group_by_seed(results):
    """按数据集+方法分组，收集不同种子的结果"""
    groups = defaultdict(lambda: {'avg_acc': [], 'forgetting': [], 'times': []})

    for r in results:
        key = (r['dataset'], r['method'])
        groups[key]['avg_acc'].append(r['avg_acc'])
        groups[key]['forgetting'].append(r['forgetting'])
        groups[key]['times'].append(r['time'])

    return groups


def print_stats(values):
    """计算统计信息"""
    if not values:
        return "N/A"

    mean = sum(values) / len(values)
    if len(values) == 1:
        return f"{mean:.2f}"

    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    std = variance ** 0.5
    return f"{mean:.2f} ± {std:.2f}"


def print_table(groups, datasets, methods):
    """打印结果表格"""
    print("\n" + "="*80)
    print("平均准确率 (Average Accuracy %)")
    print("="*80)

    header = f"{'Method':<15} " + " ".join(f"{ds:<20}" for ds in datasets)
    print(header)
    print("-"*80)

    for method in methods:
        row = f"{method:<15} "
        for dataset in datasets:
            key = (dataset, method)
            if key in groups:
                acc_stats = print_stats(groups[key]['avg_acc'])
                row += f"{acc_stats:<20} "
            else:
                row += f"{'N/A':<20} "
        print(row)

    print("\n" + "="*80)
    print("遗忘度 (Forgetting %，越低越好)")
    print("="*80)

    print(header)
    print("-"*80)

    for method in methods:
        row = f"{method:<15} "
        for dataset in datasets:
            key = (dataset, method)
            if key in groups:
                forget_stats = print_stats(groups[key]['forgetting'])
                row += f"{forget_stats:<20} "
            else:
                row += f"{'N/A':<20} "
        print(row)

    print("\n" + "="*80)
    print("运行时间 (小时)")
    print("="*80)

    print(header)
    print("-"*80)

    for method in methods:
        row = f"{method:<15} "
        for dataset in datasets:
            key = (dataset, method)
            if key in groups:
                time_stats = print_stats(groups[key]['times'])
                row += f"{time_stats:<20} "
            else:
                row += f"{'N/A':<20} "
        print(row)


def print_latex_table(groups, datasets, methods):
    """打印LaTeX格式的表格（用于论文）"""
    print("\n" + "="*80)
    print("LaTeX 表格（可直接复制到论文）")
    print("="*80 + "\n")

    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Average Accuracy and Forgetting}")
    print("\\label{tab:results}")
    print("\\begin{tabular}{l|" + "c"*len(datasets) + "}")
    print("\\hline")
    print("Method & " + " & ".join(datasets) + " \\\\")
    print("\\hline")

    for method in methods:
        row_acc = method
        row_forget = method
        for dataset in datasets:
            key = (dataset, method)
            if key in groups and len(groups[key]['avg_acc']) > 0:
                acc = groups[key]['avg_acc']
                forget = groups[key]['forgetting']
                if len(acc) > 1:
                    mean_acc = sum(acc) / len(acc)
                    std_acc = (sum((x - mean_acc)**2 for x in acc) / (len(acc)-1))**0.5
                    mean_forget = sum(forget) / len(forget)
                    std_forget = (sum((x - mean_forget)**2 for x in forget) / (len(forget)-1))**0.5
                    row_acc += f" & ${mean_acc:.1f}\\pm{std_acc:.1f}$"
                    row_forget += f" & ${mean_forget:.1f}\\pm{std_forget:.1f}$"
                else:
                    row_acc += f" & ${acc[0]:.1f}$"
                    row_forget += f" & ${forget[0]:.1f}$"
            else:
                row_acc += " & --"
                row_forget += " & --"

        print(row_acc + " \\\\")
        print(row_forget + " \\\\")
        print("\\hline")

    print("\\end{tabular}")
    print("\\end{table}")


def check_completeness(groups, datasets, methods, seeds):
    """检查实验完整性"""
    print("\n" + "="*80)
    print("实验完整性检查")
    print("="*80 + "\n")

    total_expected = len(datasets) * len(methods) * len(seeds)
    total_found = sum(len(g['avg_acc']) for g in groups.values())
    missing = total_expected - total_found

    print(f"预期实验数量: {total_expected}")
    print(f"已发现结果: {total_found}")
    print(f"缺失数量: {missing}\n")

    if missing > 0:
        print("缺失的实验配置：")
        for dataset in datasets:
            for method in methods:
                key = (dataset, method)
                if key not in groups or len(groups[key]['avg_acc']) < len(seeds):
                    found = len(groups[key]['avg_acc']) if key in groups else 0
                    print(f"  - {method} @ {dataset}: {found}/{len(seeds)} seeds")
    else:
        print("✓ 所有实验已完成！")


def main():
    # 加载结果
    results = load_results()

    if not results:
        print("未找到任何实验结果")
        return 1

    print(f"已加载 {len(results)} 个实验结果")

    # 按种子分组
    groups = group_by_seed(results)

    # 数据集和方法
    datasets = [
    'cifar-bs-50',
    'imb-cifar-bs-50',
    'noise-cifar-bs-50',
    'cifar10-bs-50',
    'imb-cifar10-bs-50',
    'noise-cifar10-bs-50',
    'mnist-bs-50',
    'imb-mnist-bs-50',
    'noise-mnist-bs-50',
]
    methods = ['bcsr', 'herding', 'gradmatch', 'gss', 'uniform']

    # 打印表格
    print_table(groups, datasets, methods)

    # 打印LaTeX表格
    print_latex_table(groups, datasets, methods)

    # 检查完整性（假设3个种子）
    check_completeness(groups, datasets, methods, seeds=[0, 1, 2])

    return 0


if __name__ == '__main__':
    sys.exit(main())
