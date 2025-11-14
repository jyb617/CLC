"""
图对比学习性能测试脚本

测试内容：
1. CUDA加速效果
2. 不同batch size的性能
3. 图构建时间
4. 前向传播时间
5. 内存占用
"""

import torch
import numpy as np
import time
from graph_features import build_graph_features, GraphFeatureGenerator, GraphContrastiveLoss
from model_CLCRec_Graph import CLCRec_Graph


def generate_synthetic_data(num_user=1000, num_item=5000, num_interactions=50000):
    """生成合成测试数据"""
    users = np.random.randint(0, num_user, num_interactions)
    items = np.random.randint(num_user, num_user + num_item, num_interactions)
    train_data = np.stack([users, items], axis=1)
    return train_data


def test_graph_building_time():
    """测试图构建时间"""
    print("=" * 80)
    print("测试1: 图构建时间")
    print("=" * 80)

    configs = [
        (1000, 5000, 50000),
        (5000, 10000, 200000),
        (10000, 20000, 500000),
    ]

    for num_user, num_item, num_interactions in configs:
        print(f"\n配置: {num_user} 用户, {num_item} 物品, {num_interactions} 交互")

        # 生成数据
        train_data = generate_synthetic_data(num_user, num_item, num_interactions)

        # 测试构建时间
        start_time = time.time()
        graph_generator = build_graph_features(train_data, num_user, num_item, device=torch.device('cuda'))
        build_time = time.time() - start_time

        print(f"  构建时间: {build_time:.3f} 秒")
        print(f"  用户-物品边数: {graph_generator.edge_index_ui.size(1)}")
        print(f"  用户-用户边数: {graph_generator.user_user_edges.size(1)}")


def test_aggregation_speed():
    """测试特征聚合速度"""
    print("\n" + "=" * 80)
    print("测试2: 特征聚合速度（CUDA加速）")
    print("=" * 80)

    num_user = 10000
    num_item = 20000
    num_interactions = 500000
    embedding_dim = 64

    # 生成数据和图
    train_data = generate_synthetic_data(num_user, num_item, num_interactions)
    graph_generator = build_graph_features(train_data, num_user, num_item, device=torch.device('cuda'))

    # 生成随机嵌入
    user_embedding = torch.randn(num_user, embedding_dim).cuda()
    item_embedding = torch.randn(num_item, embedding_dim).cuda()

    # 预热GPU
    for _ in range(10):
        _ = graph_generator(user_embedding, item_embedding)

    # 测试聚合速度
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        user_neighbor_feat, user_cooccur_feat = graph_generator(user_embedding, item_embedding)
    total_time = time.time() - start_time

    print(f"\n运行 {num_runs} 次聚合:")
    print(f"  总时间: {total_time:.3f} 秒")
    print(f"  平均时间: {total_time/num_runs*1000:.2f} 毫秒/次")
    print(f"  吞吐量: {num_runs/total_time:.1f} 次/秒")


def test_contrastive_loss_speed():
    """测试对比损失计算速度"""
    print("\n" + "=" * 80)
    print("测试3: 对比损失计算速度")
    print("=" * 80)

    batch_sizes = [128, 256, 512, 1024]
    embedding_dim = 64
    temperature = 0.2

    loss_fn = GraphContrastiveLoss(temperature=temperature)

    for batch_size in batch_sizes:
        # 生成随机特征
        view1 = torch.randn(batch_size, embedding_dim).cuda()
        view2 = torch.randn(batch_size, embedding_dim).cuda()

        # 预热
        for _ in range(10):
            _ = loss_fn(view1, view2)

        # 测试
        num_runs = 100
        start_time = time.time()
        for _ in range(num_runs):
            loss = loss_fn(view1, view2)
            # 模拟反向传播
            if torch.is_grad_enabled():
                pass
        total_time = time.time() - start_time

        print(f"\nBatch size = {batch_size}:")
        print(f"  平均时间: {total_time/num_runs*1000:.2f} 毫秒/次")
        print(f"  吞吐量: {batch_size * num_runs / total_time:.1f} 样本/秒")


def test_end_to_end_speed():
    """测试端到端前向传播速度"""
    print("\n" + "=" * 80)
    print("测试4: 端到端前向传播速度")
    print("=" * 80)

    num_user = 5000
    num_item = 10000
    num_interactions = 200000
    embedding_dim = 64
    batch_size = 256
    num_neg = 512

    # 生成数据
    train_data = generate_synthetic_data(num_user, num_item, num_interactions)
    graph_generator = build_graph_features(train_data, num_user, num_item, device=torch.device('cuda'))

    # 创建模型
    model = CLCRec_Graph(
        num_user, num_item, num_item, train_data,
        reg_weight=0.1, dim_E=embedding_dim, v_feat=None, a_feat=None, t_feat=None,
        temp_value=1.0, num_neg=num_neg, lr_lambda=1.0, is_word=False,
        num_sample=0.5, graph_temp=0.2, graph_lambda=0.1
    ).cuda()
    model.set_graph_generator(graph_generator)
    model.eval()

    # 生成批数据
    user_tensor = torch.randint(0, num_user, (batch_size, num_neg+1)).cuda()
    item_tensor = torch.randint(num_user, num_user+num_item, (batch_size, num_neg+1)).cuda()

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(user_tensor, item_tensor)

    # 测试
    num_runs = 50
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            loss, reg_loss = model(user_tensor, item_tensor)
    total_time = time.time() - start_time

    print(f"\n配置:")
    print(f"  用户数: {num_user}")
    print(f"  物品数: {num_item}")
    print(f"  Batch size: {batch_size}")
    print(f"  负样本数: {num_neg}")
    print(f"\n性能:")
    print(f"  平均时间: {total_time/num_runs*1000:.2f} 毫秒/batch")
    print(f"  吞吐量: {batch_size * num_runs / total_time:.1f} 样本/秒")


def test_memory_usage():
    """测试内存占用"""
    print("\n" + "=" * 80)
    print("测试5: 内存占用分析")
    print("=" * 80)

    num_user = 10000
    num_item = 20000
    num_interactions = 500000
    embedding_dim = 64

    # 初始内存
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1024**2

    # 构建图
    train_data = generate_synthetic_data(num_user, num_item, num_interactions)
    graph_generator = build_graph_features(train_data, num_user, num_item, device=torch.device('cuda'))
    graph_memory = torch.cuda.memory_allocated() / 1024**2

    # 创建嵌入
    user_embedding = torch.randn(num_user, embedding_dim).cuda()
    item_embedding = torch.randn(num_item, embedding_dim).cuda()
    embedding_memory = torch.cuda.memory_allocated() / 1024**2

    # 执行聚合
    user_neighbor_feat, user_cooccur_feat = graph_generator(user_embedding, item_embedding)
    aggregation_memory = torch.cuda.memory_allocated() / 1024**2

    print(f"\n内存占用:")
    print(f"  初始: {initial_memory:.2f} MB")
    print(f"  图结构: {graph_memory - initial_memory:.2f} MB")
    print(f"  嵌入: {embedding_memory - graph_memory:.2f} MB")
    print(f"  聚合后: {aggregation_memory - embedding_memory:.2f} MB")
    print(f"  总计: {aggregation_memory:.2f} MB")

    # 清理
    torch.cuda.empty_cache()


def test_scalability():
    """测试可扩展性"""
    print("\n" + "=" * 80)
    print("测试6: 可扩展性测试")
    print("=" * 80)

    embedding_dim = 64
    base_num_user = 1000

    print("\n随用户数量扩展:")
    print(f"{'用户数':<10} {'物品数':<10} {'交互数':<10} {'构建时间(s)':<15} {'聚合时间(ms)':<15}")
    print("-" * 70)

    for scale in [1, 2, 5, 10]:
        num_user = base_num_user * scale
        num_item = 5000 * scale
        num_interactions = 50000 * scale

        # 生成数据
        train_data = generate_synthetic_data(num_user, num_item, num_interactions)

        # 测试构建时间
        start = time.time()
        graph_generator = build_graph_features(train_data, num_user, num_item, device=torch.device('cuda'))
        build_time = time.time() - start

        # 测试聚合时间
        user_embedding = torch.randn(num_user, embedding_dim).cuda()
        item_embedding = torch.randn(num_item, embedding_dim).cuda()

        start = time.time()
        for _ in range(10):
            _ = graph_generator(user_embedding, item_embedding)
        agg_time = (time.time() - start) / 10 * 1000

        print(f"{num_user:<10} {num_item:<10} {num_interactions:<10} {build_time:<15.3f} {agg_time:<15.2f}")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("图对比学习性能测试套件")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("\n警告: CUDA不可用，测试将在CPU上运行（速度会很慢）")
        return

    device_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {device_name}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")

    try:
        test_graph_building_time()
        test_aggregation_speed()
        test_contrastive_loss_speed()
        test_end_to_end_speed()
        test_memory_usage()
        test_scalability()

        print("\n" + "=" * 80)
        print("所有测试完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_tests()
