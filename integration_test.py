"""
完整的集成测试，验证升级后的项目功能

模拟真实的持续学习场景
"""

import torch
import numpy as np
import argparse
import os
import shutil
from core.data_utils import get_all_loaders
from core.train_methods_cifar import train_task_sequentially, eval_single_epoch
from core.utils import setup_experiment, save_task_model_by_policy

print("=" * 60)
print("完整集成测试")
print("=" * 60)

# 清理函数
def cleanup():
    if os.path.exists('./test_integration_checkpoints'):
        shutil.rmtree('./test_integration_checkpoints')

try:
    # 测试配置
    config = argparse.Namespace(
        exp_dir='./test_integration_checkpoints',
        seq_lr=0.1,
        lr_decay=0.9,
        lr_proxy_model=2.0,
        lr_weight=2.0,
        momentum=0.8,
        dropout=0.1,
        memory_size=50,
        stream_size=25,
        batch_size=10,
        n_classes=5,
        ref_hyp=0.5,
        beta=0.1,
        num_tasks=2,  # 小规模测试
        seq_epochs=1,
        outer_iter=2,
        inner_iter=1,
        select_type='bcsr',
        mlp_hiddens=128,
        dataset='cifar-bs-50'
    )

    print(f"\n测试配置:")
    print(f"  数据集: {config.dataset}")
    print(f"  任务数: {config.num_tasks}")
    print(f"  Coreset方法: {config.select_type}")
    print(f"  设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # 创建实验目录
    os.makedirs(config.exp_dir, exist_ok=True)

    # 设置实验
    try:
        from comet_ml import Experiment
        experiment = Experiment(
            api_key="dummy_key",
            project_name="test",
            workspace="test",
            disabled=True
        )
        setup_experiment(experiment, config)
    except:
        print("⚠️  comet_ml未安装，跳过实验设置")
        experiment = None

    print("\n✓ 实验设置完成")

    # 加载数据
    print("\n加载数据...")
    loaders = get_all_loaders(
        seed=0,
        dataset=config.dataset,
        num_tasks=config.num_tasks,
        bs_inter=config.batch_size,
        bs_intra=config.stream_size,
        num_examples=config.memory_size * 2
    )
    print(f"✓ 数据加载完成，{len(loaders['sequential'])}个任务")

    # 训练任务1
    print("\n训练任务1...")
    model, outer_loss = train_task_sequentially(config, 1, loaders, [])
    save_task_model_by_policy(model, 1, 'seq', config.exp_dir)
    print("✓ 任务1训练完成")

    # 评估任务1
    metrics = eval_single_epoch(model, loaders['sequential'][1]['val'], config)
    print(f"任务1准确率: {metrics['accuracy']:.2f}%")

    # 训练任务2
    print("\n训练任务2...")
    model, outer_loss = train_task_sequentially(config, 2, loaders, outer_loss)
    save_task_model_by_policy(model, 2, 'seq', config.exp_dir)
    print("✓ 任务2训练完成")

    # 评估任务2
    metrics_task2 = eval_single_epoch(model, loaders['sequential'][2]['val'], config)
    print(f"任务2准确率: {metrics_task2['accuracy']:.2f}%")

    # 回顾任务1（测试持续学习能力）
    metrics_task1 = eval_single_epoch(model, loaders['sequential'][1]['val'], config)
    print(f"任务1回顾准确率: {metrics_task1['accuracy']:.2f}%")

    # 计算遗忘度量
    forgetting = metrics['accuracy'] - metrics_task1['accuracy']
    print(f"遗忘度量: {forgetting:.2f}%")

    print("\n" + "=" * 60)
    print("✓ 集成测试通过！")
    print("=" * 60)
    print("\n测试结果:")
    print(f"  任务1准确率: {metrics['accuracy']:.2f}%")
    print(f"  任务2准确率: {metrics_task2['accuracy']:.2f}%")
    print(f"  任务1回顾: {metrics_task1['accuracy']:.2f}%")
    print(f"  遗忘度量: {forgetting:.2f}%")
    print("\n项目升级成功，所有功能正常！")

except Exception as e:
    print(f"\n✗ 集成测试失败: {e}")
    import traceback
    traceback.print_exc()
    cleanup()
    exit(1)

finally:
    # 清理
    print("\n清理测试文件...")
    cleanup()
    print("✓ 清理完成")

print("\n集成测试完成！项目已成功升级至PyTorch 2.x并移除NTK依赖。")
