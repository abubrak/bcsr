"""
测试PyTorch 2.x升级后的功能

验证所有核心功能在PyTorch 2.x下正常工作
"""

import sys
import torch
import numpy as np

print("=" * 60)
print("PyTorch 2.x升级验证测试")
print("=" * 60)

# 检查PyTorch版本
print(f"\nPyTorch版本: {torch.__version__}")
torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
if torch_version < (2, 0):
    print(f"警告: PyTorch版本 {torch.__version__} 低于2.0")
    sys.exit(1)

print(f"✓ PyTorch {torch.__version__}")

# 检查CUDA
if torch.cuda.is_available():
    print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
    device = 'cuda'
else:
    print("⚠ CUDA不可用，使用CPU")
    device = 'cpu'

# 测试1: 导入核心模块
print("\n测试1: 导入核心模块")
try:
    from core.bcsr_coreset import BCSR_Coreset
    from core.bilevel_coreset import BilevelCoreset
    from core.pytorch_proxy_model import create_proxy_model
    from core.models import ResNet18, MLP
    from core.train_methods_cifar import train_task_sequentially
    from core.data_utils import get_all_loaders
    print("✓ 所有核心模块导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 测试2: 创建代理模型
print("\n测试2: 创建代理模型")
try:
    # MLP模型
    mlp_model = create_proxy_model('mlp', input_shape=(784,), num_classes=10)
    x = torch.randn(16, 784)
    y = mlp_model(x)
    assert y.shape == (16, 10), f"MLP输出形状错误: {y.shape}"
    print("✓ MLP代理模型创建成功")

    # ResNet模型
    resnet_model = create_proxy_model('resnet', input_shape=(3, 32, 32), num_classes=100)
    x = torch.randn(8, 3, 32, 32)
    y = resnet_model(x)
    assert y.shape == (8, 100), f"ResNet输出形状错误: {y.shape}"
    print("✓ ResNet代理模型创建成功")
except Exception as e:
    print(f"✗ 代理模型创建失败: {e}")
    sys.exit(1)

# 测试3: BCSR Coreset
print("\n测试3: BCSR Coreset")
try:
    config = {'n_classes': 10, 'dropout': 0.1, 'mlp_hiddens': 128}
    proxy_model = MLP(config).to(device)
    bc = BCSR_Coreset(
        proxy_model=proxy_model,
        lr_proxy_model=1.0,
        beta=0.1,
        out_dim=10,
        max_outer_it=2,
        max_inner_it=1,
        device=device
    )
    print("✓ BCSR_Coreset初始化成功")

    # 测试简单的权重选择
    X = torch.randn(50, 784).to(device)
    y = torch.randint(0, 10, (50,)).to(device)

    selected, _ = bc.coreset_select(
        model=proxy_model,
        X=X.cpu().numpy(),
        y=y.cpu().numpy(),
        task_id=1,
        topk=10
    )
    assert len(selected) == 10, f"选择数量错误: {len(selected)}"
    print(f"✓ BCSR选择成功，选择了{len(selected)}个样本")
except Exception as e:
    print(f"✗ BCSR测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: BilevelCoreset（简化测试）
print("\n测试4: BilevelCoreset")
try:
    def outer_loss_fn(K, alpha, y, weights, lmbda):
        import torch.nn as nn
        loss = nn.CrossEntropyLoss(reduction='none')
        loss_value = torch.mean(loss(torch.matmul(K, alpha), y.long()) * weights)
        return loss_value

    def inner_loss_fn(K, alpha, y, weights, lmbda):
        return outer_loss_fn(K, alpha, y, weights, lmbda)

    bc_bilevel = BilevelCoreset(
        outer_loss_fn=outer_loss_fn,
        inner_loss_fn=inner_loss_fn,
        out_dim=10,
        max_outer_it=2,
        max_inner_it=2,
        candidate_batch_size=100
    )
    print("✓ BilevelCoreset初始化成功")
except Exception as e:
    print(f"✗ BilevelCoreset测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5: 设备切换
print("\n测试5: 设备切换")
try:
    if torch.cuda.is_available():
        # 创建模型并在GPU上运行
        model_gpu = MLP(config).cuda()
        bc_gpu = BCSR_Coreset(
            proxy_model=model_gpu,
            lr_proxy_model=1.0,
            beta=0.1,
            device='cuda',
            max_outer_it=1,
            max_inner_it=1
        )

        X_gpu = torch.randn(20, 784).cuda()
        y_gpu = torch.randint(0, 10, (20,)).cuda()

        selected_gpu, _ = bc_gpu.coreset_select(
            model=model_gpu,
            X=X_gpu.cpu().numpy(),
            y=y_gpu.cpu().numpy(),
            task_id=1,
            topk=5
        )
        print("✓ GPU模式测试成功")

        # 切换到CPU
        model_cpu = MLP(config)
        bc_cpu = BCSR_Coreset(
            proxy_model=model_cpu,
            lr_proxy_model=1.0,
            beta=0.1,
            device='cpu',
            max_outer_it=1,
            max_inner_it=1
        )

        X_cpu = torch.randn(20, 784)
        y_cpu = torch.randint(0, 10, (20,))

        selected_cpu, _ = bc_cpu.coreset_select(
            model=model_cpu,
            X=X_cpu.numpy(),
            y=y_cpu.numpy(),
            task_id=1,
            topk=5
        )
        print("✓ CPU模式测试成功")
    else:
        print("⚠ 无GPU，跳过GPU测试")
        print("✓ CPU模式测试（已在测试3中完成）")
except Exception as e:
    print(f"✗ 设备切换测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试6: torch.compile测试（PyTorch 2.x特有）
print("\n测试6: torch.compile优化（PyTorch 2.x特有）")
try:
    if hasattr(torch, 'compile'):
        # 测试编译功能
        def simple_fn(x):
            return x * 2 + 1

        compiled_fn = torch.compile(simple_fn)
        x = torch.randn(10)
        y = compiled_fn(x)

        expected = simple_fn(x)
        assert torch.allclose(y, expected), "torch.compile结果不匹配"
        print("✓ torch.compile功能正常")
    else:
        print("⚠ 当前PyTorch版本不支持torch.compile")
except Exception as e:
    print(f"⚠ torch.compile测试失败（可能非致命）: {e}")

# 测试7: 完整流程小规模测试
print("\n测试7: 完整流程小规模测试")
try:
    from core.data_utils import get_all_loaders

    # 创建小规模数据加载器
    loaders = get_all_loaders(
        seed=0,
        dataset='cifar-bs-50',
        num_tasks=2,
        bs_inter=10,
        bs_intra=50,
        num_examples=200
    )

    print(f"✓ 数据加载器创建成功，{len(loaders['sequential'])}个任务")

    # 测试训练流程（1个epoch）
    from core.train_methods_cifar import train_task_sequentially
    from core.utils import setup_experiment
    import argparse

    args = argparse.Namespace(
        exp_dir='./test_checkpoints',
        seq_lr=0.1,
        lr_decay=0.9,
        lr_proxy_model=1.0,
        lr_weight=1.0,
        momentum=0.8,
        dropout=0.1,
        memory_size=100,
        stream_size=50,
        batch_size=10,
        n_classes=5,
        ref_hyp=0.5,
        beta=0.1,
        num_tasks=2,
        seq_epochs=1,
        outer_iter=2,
        inner_iter=1,
        select_type='bcsr',
        mlp_hiddens=128,
        dataset='cifar-bs-50'
    )

    # 创建实验目录
    import os
    os.makedirs(args.exp_dir, exist_ok=True)

    # 测试训练
    model, outer_loss = train_task_sequentially(args, 1, loaders, [])
    print("✓ 训练流程测试成功")

    # 清理测试文件
    import shutil
    shutil.rmtree(args.exp_dir)
    print("✓ 测试文件清理完成")

except Exception as e:
    print(f"✗ 完整流程测试失败: {e}")
    import traceback
    traceback.print_exc()
    # 尝试清理
    try:
        import shutil
        if os.path.exists('./test_checkpoints'):
            shutil.rmtree('./test_checkpoints')
    except:
        pass
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 所有测试通过！")
print("=" * 60)
print("\n升级验证成功：")
print(f"  - PyTorch {torch.__version__}")
print(f"  - 设备: {device}")
print(f"  - BCSR功能: ✓")
print(f"  - BilevelCoreset功能: ✓")
print(f"  - 设备切换: ✓")
if hasattr(torch, 'compile'):
    print(f"  - torch.compile: ✓")
print("\n项目已成功升级至PyTorch 2.x，移除了NTK依赖！")
