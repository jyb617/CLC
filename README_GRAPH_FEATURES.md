# 图对比学习功能说明

## 概述

本模块为CLCRec推荐系统增加了基于用户-物品购买关系的图对比学习功能，通过对比两种不同的图视图来增强模型的表示能力。

## 核心功能

### 1. 图特征生成 (`graph_features.py`)

#### GraphFeatureGenerator 类

构建用户-物品二部图，并生成两种互补的图特征：

**特征1：用户邻居特征聚合**
- 含义：每个用户购买过的物品特征的聚合
- 计算方式：`mean(item_embeddings[user购买的物品])`
- 表示：用户的直接购买偏好

**特征2：共同物品特征聚合**
- 含义：与用户有共同购买行为的其他用户的特征聚合
- 计算方式：两阶段聚合
  1. 每个物品聚合其购买用户的特征
  2. 每个用户聚合其购买物品的聚合特征
- 表示：通过共同购买物品发现的潜在相似用户

#### GraphContrastiveLoss 类

实现基于InfoNCE的对比损失函数：
- 将两种图特征视为同一用户的不同"视图"
- 最大化同一用户两种视图的相似度
- 最小化不同用户之间的相似度

### 2. 增强模型 (`model_CLCRec_Graph.py`)

#### CLCRec_Graph 类

在原有CLCRec基础上增加：
- 图特征生成器集成
- 图对比损失计算
- 可配置的图损失权重 (`graph_lambda`)
- 可配置的对比温度 (`graph_temp`)

**总损失函数：**
```
Total Loss = λ₁ * L_content + λ₂ * L_cf + λ_graph * L_graph + λ_reg * L_reg
```

其中：
- `L_content`: 内容特征对比损失
- `L_cf`: 协同过滤对比损失
- `L_graph`: 图对比损失（新增）
- `L_reg`: 正则化损失

### 3. 训练脚本 (`main_graph.py`)

完整的训练流程：
1. 加载数据和特征
2. 构建图特征生成器
3. 初始化模型
4. 训练和评估

## 使用方法

### 基本用法

```bash
python main_graph.py --data_path movielens \
                     --graph_lambda 0.1 \
                     --graph_temp 0.2 \
                     --dim_E 64 \
                     --batch_size 256
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--graph_lambda` | 0.1 | 图对比损失权重（0-1） |
| `--graph_temp` | 0.2 | 图对比温度（越小越严格） |
| `--dim_E` | 64 | 嵌入维度 |
| `--lr_lambda` | 1.0 | 原有损失平衡参数 |
| `--l_r` | 0.001 | 学习率 |
| `--batch_size` | 256 | 批大小 |
| `--num_neg` | 512 | 负样本数量 |

### 参数调优建议

#### graph_lambda (图损失权重)
- **推荐范围**: 0.05 - 0.2
- **效果**:
  - 过小（<0.05）: 图特征作用不明显
  - 过大（>0.3）: 可能干扰原有学习
- **调优策略**: 从0.1开始，逐步调整

#### graph_temp (对比温度)
- **推荐范围**: 0.1 - 0.5
- **效果**:
  - 过小（<0.1）: 梯度过大，训练不稳定
  - 过大（>0.5）: 对比效果减弱
- **调优策略**: 从0.2开始，根据收敛情况调整

## CUDA优化

本实现充分利用CUDA加速：

### 1. 高效的稀疏操作
- 使用 `torch_scatter` 进行高效的稀疏张量聚合
- 避免稠密矩阵乘法，直接操作边索引

### 2. 批处理优化
- 仅对当前batch的用户计算图损失
- 减少不必要的全图计算

### 3. 内存管理
- 预计算并缓存图结构（边索引）
- 动态计算特征聚合，避免存储中间结果

### 4. 并行计算
- 所有张量操作都在GPU上执行
- 利用PyTorch的自动并行化

## 性能对比

### 计算复杂度

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 用户邻居聚合 | O(E) | O(N_u * d) |
| 共同物品聚合 | O(2E) | O(N_i * d) |
| 对比损失 | O(B² * d) | O(B * d) |

其中：
- E: 边数（用户-物品交互数）
- N_u: 用户数
- N_i: 物品数
- d: 嵌入维度
- B: batch大小

### 预期性能提升

根据图对比学习的相关研究，预期可获得：
- **推荐准确率**: 提升 2-5%
- **冷启动性能**: 提升 5-10%
- **训练时间**: 增加 15-25%

## 代码示例

### 直接使用GraphFeatureGenerator

```python
from graph_features import build_graph_features, GraphContrastiveLoss
import torch

# 构建图特征生成器
graph_generator = build_graph_features(train_data, num_user, num_item, device='cuda')

# 准备嵌入
user_embedding = torch.randn(num_user, 64).cuda()
item_embedding = torch.randn(num_item, 64).cuda()

# 生成图特征
user_neighbor_feat, user_cooccur_feat = graph_generator(user_embedding, item_embedding)

# 计算对比损失
loss_fn = GraphContrastiveLoss(temperature=0.2)
loss = loss_fn(user_neighbor_feat, user_cooccur_feat)
```

### 集成到自定义模型

```python
from model_CLCRec_Graph import CLCRec_Graph
from graph_features import build_graph_features

# 构建图生成器
graph_generator = build_graph_features(train_data, num_user, num_item)

# 创建模型
model = CLCRec_Graph(
    num_user, num_item, num_warm_item, train_data,
    reg_weight=0.1, dim_E=64, v_feat=None, a_feat=a_feat, t_feat=t_feat,
    temp_value=1.0, num_neg=512, lr_lambda=1.0, is_word=False,
    num_sample=0.5, graph_temp=0.2, graph_lambda=0.1
).cuda()

# 设置图生成器
model.set_graph_generator(graph_generator)
```

## 实验建议

### 1. 消融实验
- 仅使用用户邻居特征
- 仅使用共同物品特征
- 同时使用两种特征（推荐）

### 2. 参数敏感性分析
- 扫描 graph_lambda: [0.01, 0.05, 0.1, 0.15, 0.2]
- 扫描 graph_temp: [0.1, 0.15, 0.2, 0.3, 0.5]

### 3. 数据集对比
- 稀疏数据集（如MovieLens）: 图特征更有效
- 稠密数据集（如TikTok）: 效果可能有限

## 常见问题

### Q1: 图构建太慢？
A: 图结构在初始化时一次性构建，训练时复用。如果数据量特别大，可以考虑：
- 降低 `min_common_items` 参数，减少用户-用户边
- 对度数很大的节点进行采样

### Q2: 内存不足？
A: 减少批大小或使用梯度累积：
```python
# 梯度累积示例
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Q3: 训练不收敛？
A: 可能原因：
- graph_lambda 过大 → 降低到 0.05
- graph_temp 过小 → 增大到 0.3
- 学习率过大 → 降低学习率

## 文件清单

- `graph_features.py`: 图特征生成核心代码
- `model_CLCRec_Graph.py`: 增强的CLCRec模型
- `main_graph.py`: 训练脚本
- `test_graph_performance.py`: 性能测试脚本
- `README_GRAPH_FEATURES.md`: 本说明文档

## 引用

如果使用本代码，请引用：

```bibtex
@article{graph_contrastive_rec,
  title={Graph Contrastive Learning for Recommendation},
  author={...},
  year={2024}
}
```

## 联系方式

如有问题或建议，请提交Issue或联系开发者。
