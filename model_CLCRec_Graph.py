from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter
from graph_features import GraphFeatureGenerator, GraphContrastiveLoss

##########################################################################

class CLCRec_Graph(torch.nn.Module):
    """
    CLCRec with Graph Contrastive Learning

    在原有CLCRec基础上增加：
    1. 用户-物品二部图特征
    2. 用户邻居特征聚合（购买的物品）
    3. 共现特征聚合（共同购买用户）
    4. 图对比损失函数
    """

    def __init__(self, num_user, num_item, num_warm_item, edge_index, reg_weight,
                 dim_E, v_feat, a_feat, t_feat, temp_value, num_neg, lr_lambda,
                 is_word, num_sample=0.5, graph_temp=0.2, graph_lambda=0.1):
        super(CLCRec_Graph, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.num_warm_item = num_warm_item
        self.num_neg = num_neg
        self.lr_lambda = lr_lambda
        self.reg_weight = reg_weight
        self.temp_value = temp_value
        self.dim_E = dim_E
        self.is_word = is_word
        self.num_sample = num_sample

        # 图对比学习参数
        self.graph_lambda = graph_lambda  # 图对比损失权重
        self.graph_temp = graph_temp  # 图对比温度

        # ID嵌入矩阵
        self.id_embedding = nn.Parameter(
            nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_E)))
        )

        # 多模态特征处理
        self.dim_feat = 0

        if v_feat is not None:
            self.v_feat = F.normalize(v_feat, dim=1)
            self.dim_feat += self.v_feat.size(1)
        else:
            self.v_feat = None

        if a_feat is not None:
            self.a_feat = F.normalize(a_feat, dim=1)
            self.dim_feat += self.a_feat.size(1)
        else:
            self.a_feat = None

        if t_feat is not None:
            if is_word:
                self.t_feat = nn.Parameter(
                    nn.init.xavier_normal_(torch.rand((torch.max(t_feat[1]).item()+1, 128)))
                )
                self.word_tensor = t_feat
            else:
                self.t_feat = F.normalize(t_feat, dim=1)
            self.dim_feat += self.t_feat.size(1)
        else:
            self.t_feat = None

        # 编码器层
        self.MLP = nn.Linear(dim_E, dim_E)
        self.encoder_layer1 = nn.Linear(self.dim_feat, 256)
        self.encoder_layer2 = nn.Linear(256, dim_E)

        # 注意力层
        self.att_weight_1 = nn.Parameter(nn.init.kaiming_normal_(torch.rand((dim_E, dim_E))))
        self.att_weight_2 = nn.Parameter(nn.init.kaiming_normal_(torch.rand((dim_E, dim_E))))
        self.bias = nn.Parameter(nn.init.kaiming_normal_(torch.rand((dim_E, 1))))
        self.att_sum_layer = nn.Linear(dim_E, dim_E)

        self.result = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_E))).cuda()

        # 图特征生成器（稍后初始化）
        self.graph_generator = None

        # 图对比损失
        self.graph_contrastive_loss_fn = GraphContrastiveLoss(temperature=graph_temp)

    def set_graph_generator(self, graph_generator: GraphFeatureGenerator):
        """设置图特征生成器"""
        self.graph_generator = graph_generator

    def encoder(self, mask=None):
        """将多模态内容特征编码为统一的嵌入表示"""
        # 如果没有任何特征，返回零特征
        if self.dim_feat == 0:
            return torch.zeros(self.num_item, self.dim_E).cuda()

        feature = torch.tensor([]).cuda()

        if self.v_feat is not None:
            feature = torch.cat((feature, self.v_feat), dim=1)

        if self.a_feat is not None:
            feature = torch.cat((feature, self.a_feat), dim=1)

        if self.t_feat is not None:
            if self.is_word:
                t_feat = F.normalize(
                    scatter(self.t_feat[self.word_tensor[1]],
                           self.word_tensor[0], dim=0, reduce='mean')
                ).cuda()
                feature = torch.cat((feature, t_feat), dim=1)
            else:
                feature = torch.cat((feature, self.t_feat), dim=1)

        feature = F.leaky_relu(self.encoder_layer1(feature))
        feature = self.encoder_layer2(feature)
        return feature

    def loss_contrastive(self, tensor_anchor, tensor_all, temp_value):
        """原有的对比损失"""
        all_score = torch.exp(torch.sum(tensor_anchor*tensor_all, dim=1)/temp_value).view(-1, 1+self.num_neg)
        all_score = all_score.view(-1, 1+self.num_neg)
        pos_score = all_score[:, 0]
        all_score = torch.sum(all_score, dim=1)
        self.mat = (1-pos_score/all_score).mean()
        contrastive_loss = (-torch.log(pos_score / all_score)).mean()
        return contrastive_loss

    def compute_graph_contrastive_loss(self, user_tensor: torch.Tensor):
        """
        计算图对比损失

        Args:
            user_tensor: [batch_size] 用户索引

        Returns:
            graph_loss: 图对比损失
        """
        if self.graph_generator is None:
            return torch.tensor(0.0).cuda()

        # 获取当前的用户和物品嵌入
        user_embedding = self.id_embedding[:self.num_user]  # [num_user, dim_E]
        item_embedding = self.id_embedding[self.num_user:self.num_user+self.num_item]  # [num_item, dim_E]

        # 生成图特征
        user_neighbor_feat, user_cooccur_feat = self.graph_generator(
            user_embedding, item_embedding
        )

        # 仅对batch中的用户计算损失（提高效率）
        unique_users = torch.unique(user_tensor)
        batch_neighbor_feat = user_neighbor_feat[unique_users]  # [batch_users, dim_E]
        batch_cooccur_feat = user_cooccur_feat[unique_users]  # [batch_users, dim_E]

        # 计算图对比损失
        graph_loss = self.graph_contrastive_loss_fn(
            batch_neighbor_feat, batch_cooccur_feat
        )

        return graph_loss

    def forward(self, user_tensor, item_tensor):
        """前向传播"""
        pos_item_tensor = item_tensor[:, 0].unsqueeze(1)
        pos_item_tensor = pos_item_tensor.repeat(1, 1+self.num_neg).view(-1, 1).squeeze()

        user_tensor_flat = user_tensor.view(-1, 1).squeeze()
        item_tensor_flat = item_tensor.view(-1, 1).squeeze()

        # 编码多模态特征
        feature = self.encoder()
        all_item_feat = feature[item_tensor_flat-self.num_user]

        # 获取嵌入
        user_embedding = self.id_embedding[user_tensor_flat]
        pos_item_embedding = self.id_embedding[pos_item_tensor]
        all_item_embedding = self.id_embedding[item_tensor_flat]

        # 归一化
        head_feat = F.normalize(all_item_feat, dim=1)
        head_embed = F.normalize(pos_item_embedding, dim=1)

        # 随机混合ID嵌入和内容特征
        all_item_input = all_item_embedding.clone()
        rand_index = torch.randint(
            all_item_embedding.size(0),
            (int(all_item_embedding.size(0)*self.num_sample), )
        ).cuda()
        all_item_input[rand_index] = all_item_feat[rand_index].clone()

        # 原有的对比损失
        self.contrastive_loss_1 = self.loss_contrastive(head_embed, head_feat, self.temp_value)
        self.contrastive_loss_2 = self.loss_contrastive(user_embedding, all_item_input, self.temp_value)

        # 计算图对比损失
        self.graph_contrastive_loss = self.compute_graph_contrastive_loss(user_tensor_flat)

        # 正则化损失
        reg_loss = ((torch.sqrt((user_embedding**2).sum(1))).mean() +
                   (torch.sqrt((all_item_embedding**2).sum(1))).mean()) / 2

        # 更新结果
        self.result = torch.cat((self.id_embedding[:self.num_user+self.num_warm_item],
                                feature[self.num_warm_item:]), dim=0)

        # 总对比损失 = 原有损失 + 图对比损失
        total_contrastive_loss = (self.contrastive_loss_1 * self.lr_lambda +
                                 self.contrastive_loss_2 * (1-self.lr_lambda) +
                                 self.graph_contrastive_loss * self.graph_lambda)

        return total_contrastive_loss, reg_loss

    def loss(self, user_tensor, item_tensor):
        """计算总损失"""
        contrastive_loss, reg_loss = self.forward(user_tensor, item_tensor)
        reg_loss = self.reg_weight * reg_loss
        return reg_loss + contrastive_loss, self.contrastive_loss_2 + reg_loss, reg_loss
