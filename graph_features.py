import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import random


class GraphFeatureGenerator(nn.Module):
    """
    用户-物品购买关系图特征生成器

    功能：
    1. 构建用户-物品二部图
    2. 生成用户邻居特征（用户购买过的物品特征聚合）
    3. 生成共同物品特征（与用户有共同物品的其他用户特征聚合）
    4. 充分利用CUDA加速
    """

    def __init__(self, num_user: int, num_item: int, user_item_dict: Dict,
                 device: torch.device = torch.device('cuda'),
                 max_users_per_item: int = 100,
                 enable_user_cooccurrence: bool = True):
        super(GraphFeatureGenerator, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.device = device
        self.max_users_per_item = max_users_per_item
        self.enable_user_cooccurrence = enable_user_cooccurrence

        # 构建用户-物品二部图的边索引
        self.edge_index_ui, self.edge_index_iu = self._build_bipartite_graph(user_item_dict)

        # 构建用户-用户共现矩阵（基于共同购买的物品）- 可选
        if enable_user_cooccurrence:
            print("  [注意] 构建用户共现图（可能较慢）...")
            self.user_user_edges = self._build_user_cooccurrence_graph(
                user_item_dict, max_users_per_item=max_users_per_item
            )
            print(f"\n  ✓ 图构建完成: {self.edge_index_ui.size(1):,} 条用户-物品边, "
                  f"{self.user_user_edges.size(1):,} 条用户-用户边")
        else:
            print("\n  ⚠️  用户共现图已禁用（快速模式）- 仅使用用户-物品邻居特征")
            self.user_user_edges = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            print(f"  ✓ 图构建完成: {self.edge_index_ui.size(1):,} 条用户-物品边")

    def _build_bipartite_graph(self, user_item_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建用户-物品二部图的边索引

        Returns:
            edge_index_ui: [2, num_edges] 用户->物品边
            edge_index_iu: [2, num_edges] 物品->用户边（反向）
        """
        user_list = []
        item_list = []

        print("  [1/3] 构建用户-物品二部图...")
        for user, items in tqdm(user_item_dict.items(), desc="    处理用户", ncols=80):
            user_list.extend([user] * len(items))
            item_list.extend(items)

        # 用户->物品边
        edge_index_ui = torch.tensor([user_list, item_list], dtype=torch.long, device=self.device)

        # 物品->用户边（反向）
        edge_index_iu = torch.tensor([item_list, user_list], dtype=torch.long, device=self.device)

        return edge_index_ui, edge_index_iu

    def _build_user_cooccurrence_graph(self, user_item_dict: Dict,
                                       min_common_items: int = 2,
                                       max_users_per_item: int = 100) -> torch.Tensor:
        """
        构建用户-用户共现图（基于共同购买的物品）

        Args:
            user_item_dict: 用户-物品字典
            min_common_items: 最少共同物品数量
            max_users_per_item: 每个物品最多考虑的用户数（避免热门物品计算爆炸）

        Returns:
            edge_index: [2, num_edges] 用户-用户边
        """
        # 构建物品->用户的倒排索引
        print("  [2/3] 构建物品-用户倒排索引...")
        item_users_dict = {}
        for user, items in tqdm(user_item_dict.items(), desc="    索引构建", ncols=80):
            for item in items:
                if item not in item_users_dict:
                    item_users_dict[item] = []
                item_users_dict[item].append(user)

        # 计算用户-用户共现（优化版：对热门物品采样）
        print("  [3/3] 计算用户共现关系...")
        print(f"        最多考虑每个物品的 {max_users_per_item} 个用户（避免计算爆炸）")

        user_cooccur = {}
        sampled_items = 0
        total_pairs = 0

        for item, users in tqdm(item_users_dict.items(), desc="    共现计算", ncols=80):
            # 如果用户数超过阈值，随机采样
            if len(users) > max_users_per_item:
                users = random.sample(users, max_users_per_item)
                sampled_items += 1

            # 对于每个物品，其购买用户之间两两连接
            for i, u1 in enumerate(users):
                for u2 in users[i+1:]:
                    key = (min(u1, u2), max(u1, u2))  # 确保边的唯一性
                    user_cooccur[key] = user_cooccur.get(key, 0) + 1
                    total_pairs += 1

        print(f"        已处理 {len(item_users_dict)} 个物品")
        print(f"        其中 {sampled_items} 个热门物品被采样")
        print(f"        生成 {total_pairs:,} 个候选用户对")

        # 过滤掉共同物品数量过少的边
        edges = [(u1, u2) for (u1, u2), count in user_cooccur.items()
                 if count >= min_common_items]

        if len(edges) == 0:
            # 如果没有符合条件的边，返回空边索引
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)

        # 双向边
        edges_bidirectional = edges + [(u2, u1) for u1, u2 in edges]
        edge_index = torch.tensor(edges_bidirectional, dtype=torch.long, device=self.device).t()

        return edge_index

    def aggregate_neighbor_features(self, user_embedding: torch.Tensor,
                                    item_embedding: torch.Tensor,
                                    aggr: str = 'mean') -> torch.Tensor:
        """
        聚合用户邻居特征（用户购买过的物品特征）

        Args:
            user_embedding: [num_user, dim] 用户嵌入
            item_embedding: [num_item, dim] 物品嵌入
            aggr: 聚合方式，'mean' 或 'sum'

        Returns:
            user_neighbor_feat: [num_user, dim] 用户的邻居物品特征聚合
        """
        # edge_index_ui: [2, num_edges], [0]是用户索引, [1]是物品索引
        user_idx = self.edge_index_ui[0]  # [num_edges]
        item_idx = self.edge_index_ui[1] - self.num_user  # 物品索引需要减去num_user偏移

        # 获取对应的物品嵌入
        neighbor_item_feats = item_embedding[item_idx]  # [num_edges, dim]

        # 聚合到用户
        if aggr == 'mean':
            user_neighbor_feat = scatter_mean(neighbor_item_feats, user_idx,
                                             dim=0, dim_size=self.num_user)
        else:  # sum
            user_neighbor_feat = scatter_add(neighbor_item_feats, user_idx,
                                            dim=0, dim_size=self.num_user)

        return user_neighbor_feat

    def aggregate_cooccurrence_features(self, user_embedding: torch.Tensor,
                                       item_embedding: torch.Tensor,
                                       aggr: str = 'mean') -> torch.Tensor:
        """
        聚合共同物品特征（与用户有共同购买行为的其他用户的特征）

        两阶段聚合：
        1. 物品 -> 购买该物品的用户特征聚合
        2. 用户 -> 其购买物品的聚合特征

        Args:
            user_embedding: [num_user, dim] 用户嵌入
            item_embedding: [num_item, dim] 物品嵌入
            aggr: 聚合方式

        Returns:
            user_cooccur_feat: [num_user, dim] 用户的共现特征
        """
        # 阶段1: 聚合每个物品的购买用户特征
        # edge_index_iu: [2, num_edges], [0]是物品索引, [1]是用户索引
        item_idx = self.edge_index_iu[0] - self.num_user  # 物品索引
        user_idx = self.edge_index_iu[1]  # 用户索引

        # 获取对应的用户嵌入
        user_feats = user_embedding[user_idx]  # [num_edges, dim]

        # 聚合到物品：每个物品获得其购买用户的聚合特征
        if aggr == 'mean':
            item_user_agg = scatter_mean(user_feats, item_idx,
                                        dim=0, dim_size=self.num_item)
        else:
            item_user_agg = scatter_add(user_feats, item_idx,
                                       dim=0, dim_size=self.num_item)

        # 阶段2: 将物品的聚合用户特征传播回用户
        # 使用edge_index_ui，将item_user_agg聚合到用户
        user_idx_stage2 = self.edge_index_ui[0]
        item_idx_stage2 = self.edge_index_ui[1] - self.num_user

        cooccur_feats = item_user_agg[item_idx_stage2]  # [num_edges, dim]

        if aggr == 'mean':
            user_cooccur_feat = scatter_mean(cooccur_feats, user_idx_stage2,
                                            dim=0, dim_size=self.num_user)
        else:
            user_cooccur_feat = scatter_add(cooccur_feats, user_idx_stage2,
                                           dim=0, dim_size=self.num_user)

        return user_cooccur_feat

    def forward(self, user_embedding: torch.Tensor,
                item_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：生成用户的两种图特征

        Args:
            user_embedding: [num_user, dim] 用户嵌入
            item_embedding: [num_item, dim] 物品嵌入

        Returns:
            view1: [num_user, dim] 第一个视图
            view2: [num_user, dim] 第二个视图

        Fast mode (enable_user_cooccurrence=False):
            view1 = user ID embedding
            view2 = user neighbor features (aggregated item features)

        Full mode (enable_user_cooccurrence=True):
            view1 = user neighbor features
            view2 = user co-occurrence features (two-stage aggregation)
        """
        # 用户邻居特征（直接购买的物品）
        user_neighbor_feat = self.aggregate_neighbor_features(
            user_embedding, item_embedding, aggr='mean'
        )

        if self.enable_user_cooccurrence:
            # 完整模式：使用两阶段聚合作为第二个视图
            user_cooccur_feat = self.aggregate_cooccurrence_features(
                user_embedding, item_embedding, aggr='mean'
            )
            return user_neighbor_feat, user_cooccur_feat
        else:
            # 快速模式：使用ID嵌入作为第一个视图，邻居特征作为第二个视图
            # 这样可以形成有意义的对比学习，让ID嵌入和结构特征对齐
            return user_embedding, user_neighbor_feat


class GraphContrastiveLoss(nn.Module):
    """
    图对比损失函数

    对比两种视图：
    1. 用户邻居特征综合（从用户购买的物品角度）
    2. 共同物品特征综合（从共同用户角度）
    """

    def __init__(self, temperature: float = 0.1):
        super(GraphContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, view1: torch.Tensor, view2: torch.Tensor,
                user_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算对比损失（InfoNCE）

        Args:
            view1: [batch_size, dim] 第一个视图（用户邻居特征）
            view2: [batch_size, dim] 第二个视图（共现特征）
            user_indices: [batch_size] 用户索引，用于batch采样

        Returns:
            loss: 对比损失值
        """
        # L2归一化
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)

        batch_size = view1.size(0)

        # 计算相似度矩阵: [batch_size, batch_size]
        similarity_matrix = torch.matmul(view1, view2.t()) / self.temperature

        # 对角线是正样本，其余是负样本
        # 正样本: (view1[i], view2[i])
        pos_sim = torch.diag(similarity_matrix)  # [batch_size]

        # InfoNCE损失
        # log(exp(pos) / sum(exp(all)))
        exp_sim = torch.exp(similarity_matrix)  # [batch_size, batch_size]
        exp_sum = exp_sim.sum(dim=1)  # [batch_size]

        loss = -torch.log(torch.exp(pos_sim) / exp_sum).mean()

        return loss


def build_graph_features(train_data: np.ndarray, num_user: int, num_item: int,
                        device: torch.device = torch.device('cuda'),
                        max_users_per_item: int = 100,
                        enable_user_cooccurrence: bool = True) -> GraphFeatureGenerator:
    """
    从训练数据构建图特征生成器

    Args:
        train_data: [(user, item), ...] 训练数据
        num_user: 用户数量
        num_item: 物品数量
        device: 设备
        max_users_per_item: 每个物品最多考虑的用户数，用于避免热门物品计算爆炸
                           默认100。如果数据集很大或很稀疏，可以适当增大。
        enable_user_cooccurrence: 是否启用用户共现图（禁用可大幅加速）
                                 默认True。如果构建太慢，设为False。

    Returns:
        graph_generator: 图特征生成器
    """
    # 构建用户-物品字典
    if enable_user_cooccurrence:
        print("  [0/3] 构建用户-物品字典...")
    else:
        print("  [0/1] 构建用户-物品字典（快速模式）...")

    user_item_dict = {}
    for user, item in tqdm(train_data, desc="    处理交互", ncols=80):
        user = int(user)
        item = int(item)
        if user not in user_item_dict:
            user_item_dict[user] = []
        user_item_dict[user].append(item)

    # 创建图特征生成器
    graph_generator = GraphFeatureGenerator(
        num_user, num_item, user_item_dict, device, max_users_per_item, enable_user_cooccurrence
    )

    return graph_generator
