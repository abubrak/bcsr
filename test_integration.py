"""
简单的集成测试,验证新方法是否正确集成到训练流程中
"""
import torch
import numpy as np
from core.coreset_baselines import HerdingCoreset
from core.gradmatch_coreset import GradMatchCoreset
from core.gss_coreset import GSSCoreset
from core.summary import Summarizer
import sys
import io

# 修复 Windows 控制台编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_herding():
    """测试 Herding 方法"""
    print("Testing HerdingCoreset...")
    try:
        device = 'cpu'
        herding = HerdingCoreset(device=device)

        # 创建虚拟数据
        num_samples = 100
        num_features = 32 * 32 * 3  # CIFAR-10 图像展平后的大小
        num_classes = 5

        data = torch.randn(num_samples, 3, 32, 32)
        targets = torch.randint(0, num_classes, (num_samples,))
        budget = 20
        task_id = 1

        # 创建简单的模型用于梯度计算
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(num_features, num_classes)

            def forward(self, x, task_id=None):
                x = x.view(x.size(0), -1)
                return self.fc(x)

            def embed(self, x, task_id=None):
                """提取特征"""
                x = x.view(x.size(0), -1)
                return x

        model = SimpleModel()

        # 测试 select 方法
        selected = herding.select(data, targets, budget, task_id, model=model)
        assert isinstance(selected, (list, np.ndarray, torch.Tensor)), "返回值应该是索引列表"
        assert len(selected) <= budget, f"选择的样本数 {len(selected)} 不应超过预算 {budget}"
        print(f"  ✓ HerdingCoreset select() 成功, 选择了 {len(selected)} 个样本")
        return True
    except Exception as e:
        print(f"  ✗ HerdingCoreset 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradmatch():
    """测试 GradMatch 方法"""
    print("Testing GradMatchCoreset...")
    try:
        device = 'cpu'
        gradmatch = GradMatchCoreset(device=device)

        # 创建虚拟数据
        num_samples = 100
        num_features = 32 * 32 * 3
        num_classes = 5

        data = torch.randn(num_samples, 3, 32, 32)
        targets = torch.randint(0, num_classes, (num_samples,))
        budget = 20
        task_id = 1

        # 创建简单的模型
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(num_features, num_classes)

            def forward(self, x, task_id=None):
                x = x.view(x.size(0), -1)
                return self.fc(x)

            def embed(self, x, task_id=None):
                """提取特征"""
                x = x.view(x.size(0), -1)
                return x

        model = SimpleModel()

        # 测试 select 方法
        selected = gradmatch.select(data, targets, budget, task_id, model=model)
        assert isinstance(selected, (list, np.ndarray, torch.Tensor)), "返回值应该是索引列表"
        assert len(selected) <= budget, f"选择的样本数 {len(selected)} 不应超过预算 {budget}"
        print(f"  ✓ GradMatchCoreset select() 成功, 选择了 {len(selected)} 个样本")
        return True
    except Exception as e:
        print(f"  ✗ GradMatchCoreset 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gss():
    """测试 GSS 方法"""
    print("Testing GSSCoreset...")
    try:
        device = 'cpu'
        gss = GSSCoreset(device=device)

        # 创建虚拟数据
        num_samples = 100
        num_features = 32 * 32 * 3
        num_classes = 5

        data = torch.randn(num_samples, 3, 32, 32)
        targets = torch.randint(0, num_classes, (num_samples,))
        budget = 20
        task_id = 1

        # 创建简单的模型
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(num_features, num_classes)

            def forward(self, x, task_id=None):
                x = x.view(x.size(0), -1)
                return self.fc(x)

            def embed(self, x, task_id=None):
                """提取特征"""
                x = x.view(x.size(0), -1)
                return x

        model = SimpleModel()

        # 测试 select 方法
        selected = gss.select(data, targets, budget, task_id, model=model)
        assert isinstance(selected, (list, np.ndarray, torch.Tensor)), "返回值应该是索引列表"
        assert len(selected) <= budget, f"选择的样本数 {len(selected)} 不应超过预算 {budget}"
        print(f"  ✓ GSSCoreset select() 成功, 选择了 {len(selected)} 个样本")
        return True
    except Exception as e:
        print(f"  ✗ GSSCoreset 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_factory():
    """测试 Summarizer factory 方法"""
    print("Testing Summarizer.factory()...")
    try:
        rs = np.random.RandomState(0)

        # 测试所有方法
        methods = ['uniform', 'herding', 'gradmatch', 'gss']
        for method in methods:
            summarizer = Summarizer.factory(method, rs)
            assert summarizer is not None, f"Factory 返回了 None 对于方法 {method}"
            print(f"  ✓ Factory 成功创建 {method} summarizer")

        return True
    except Exception as e:
        print(f"  ✗ Factory 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("运行集成测试")
    print("="*60)

    results = []
    results.append(("Herding", test_herding()))
    results.append(("GradMatch", test_gradmatch()))
    results.append(("GSS", test_gss()))
    results.append(("Factory", test_factory()))

    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:15s}: {status}")

    all_passed = all(result for _, result in results)
    print("="*60)
    if all_passed:
        print("所有测试通过! ✓")
    else:
        print("部分测试失败! ✗")
    print("="*60)

    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
