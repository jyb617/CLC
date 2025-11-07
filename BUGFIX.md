# 修复记录

## 2025-11-07: 修复encoder处理空特征的问题

### 问题描述
当模型没有任何多模态特征（v_feat, a_feat, t_feat都是None）时，encoder()方法会创建空张量并尝试通过Linear层处理，导致维度错误：

```
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
```

### 问题原因
1. 当所有特征都是None时，`dim_feat = 0`
2. encoder()创建空张量 `feature = torch.tensor([]).cuda()` (shape=[0])
3. 尝试通过 `encoder_layer1` (nn.Linear(0, 256)) 处理空张量导致错误

### 修复方案
在 `model_CLCRec_Graph.py` 的 `encoder()` 方法开始处添加检查：

```python
def encoder(self, mask=None):
    """将多模态内容特征编码为统一的嵌入表示"""
    # 如果没有任何特征，返回零特征
    if self.dim_feat == 0:
        return torch.zeros(self.num_item, self.dim_E).cuda()

    # ... 原有代码
```

### 影响
- 修复后模型可以在没有多模态特征时正常运行
- 性能测试脚本可以正常执行
- 不影响有特征时的正常使用

### 测试
```python
# 测试无特征情况
model = CLCRec_Graph(
    num_user, num_item, num_warm_item, train_data,
    reg_weight=0.1, dim_E=64,
    v_feat=None, a_feat=None, t_feat=None,  # 所有特征都是None
    ...
).cuda()

# 应该能正常运行
user_tensor = torch.randint(0, num_user, (batch_size, num_neg+1)).cuda()
item_tensor = torch.randint(num_user, num_user+num_item, (batch_size, num_neg+1)).cuda()
loss, reg_loss = model(user_tensor, item_tensor)  # 正常执行
```

### 文件修改
- `model_CLCRec_Graph.py`: encoder() 方法（第97-123行）
