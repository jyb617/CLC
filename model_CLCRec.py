from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter

##########################################################################
# LightGCN Encoder Module
##########################################################################

class LightGCN(nn.Module):
    """
    LightGCN encoder for collaborative filtering.
    Replaces the original id_embedding with graph convolution.
    """
    def __init__(self, num_users, num_items, embed_dim, num_layers):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Learnable embedding matrix E^(0)
        self.embedding = nn.Embedding(num_users + num_items, embed_dim)
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, graph):
        """
        Perform K-layer graph convolution.

        Args:
            graph: Normalized sparse adjacency matrix

        Returns:
            List of embeddings [E^(0), E^(1), ..., E^(K)]
        """
        all_embeddings = [self.embedding.weight]
        current_embeddings = self.embedding.weight

        for layer in range(self.num_layers):
            # LightGCN propagation: E^(k+1) = A @ E^(k)
            current_embeddings = torch.sparse.mm(graph, current_embeddings)
            all_embeddings.append(current_embeddings)

        return all_embeddings

    def get_final_embeddings(self, all_embeddings_list):
        """
        Layer combination using mean pooling.

        Args:
            all_embeddings_list: List of embeddings from all layers

        Returns:
            Final embedding E_final = mean([E^(0), E^(1), ..., E^(K)])
        """
        stacked_embeddings = torch.stack(all_embeddings_list, dim=0)
        final_embeddings = torch.mean(stacked_embeddings, dim=0)
        return final_embeddings


##########################################################################
# Structured R-E Loss Function
##########################################################################

def calc_infonce_with_mask(anchor, keys, positive_mask, temperature):
    """
    Calculate InfoNCE loss with custom positive mask.

    Args:
        anchor: Content embeddings [batch_size, dim]
        keys: Collaborative embeddings [batch_size, dim]
        positive_mask: Binary mask [batch_size, batch_size] indicating positive pairs
        temperature: Temperature parameter

    Returns:
        InfoNCE loss (scalar)
    """
    # Normalize embeddings
    anchor = F.normalize(anchor, p=2, dim=1)
    keys = F.normalize(keys, p=2, dim=1)

    # Compute similarity matrix
    sim_matrix = torch.matmul(anchor, keys.T) / temperature

    # Compute exp
    exp_sim = torch.exp(sim_matrix)

    # Numerator: sum over positives
    numerator = (exp_sim * positive_mask).sum(dim=1)

    # Denominator: sum over all
    denominator = exp_sim.sum(dim=1)

    # Avoid log(0)
    numerator = torch.clamp(numerator, min=1e-9)
    denominator = torch.clamp(denominator, min=1e-9)

    # InfoNCE loss
    loss = -torch.log(numerator / denominator).mean()
    return loss


##########################################################################

class CLCRec(torch.nn.Module):
    def __init__(self, num_user, num_item, num_warm_item, edge_index, reg_weight, dim_E, v_feat, a_feat, t_feat, num_neg, is_word,
                 # New parameters for LightGCN and R-E loss
                 num_layers=3, lambda_re=0.1, temperature_re=0.1, temperature_ui=1.0, structural_threshold=0.8):
        super(CLCRec, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.num_warm_item = num_warm_item
        self.num_neg = num_neg
        self.reg_weight = reg_weight
        self.dim_E = dim_E
        self.is_word = is_word

        # Store new parameters
        self.num_layers = num_layers
        self.lambda_re = lambda_re
        self.temperature_re = temperature_re
        self.temperature_ui = temperature_ui
        self.structural_threshold = structural_threshold

        # Replace id_embedding with LightGCN
        self.lightgcn = LightGCN(num_user, num_item, dim_E, num_layers)

        self.dim_feat = 0
        
        if v_feat is not None:
            self.v_feat = F.normalize(v_feat, dim=1)#归一化
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
                self.t_feat = nn.Parameter(nn.init.xavier_normal_(torch.rand((torch.max(t_feat[1]).item()+1, 128))))
                self.word_tensor = t_feat
            else:
                self.t_feat = F.normalize(t_feat, dim=1)
            self.dim_feat += self.t_feat.size(1)
        else:
            self.t_feat = None
        
        self.MLP = nn.Linear(dim_E, dim_E)

        self.encoder_layer1 = nn.Linear(self.dim_feat, 256)
        self.encoder_layer2 = nn.Linear(256, dim_E)
        
        self.att_weight_1 = nn.Parameter(nn.init.kaiming_normal_(torch.rand((dim_E, dim_E))))
        self.att_weight_2 = nn.Parameter(nn.init.kaiming_normal_(torch.rand((dim_E, dim_E))))
        self.bias = nn.Parameter(nn.init.kaiming_normal_(torch.rand((dim_E, 1))))
        self.att_sum_layer = nn.Linear(dim_E, dim_E)

        self.result = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_E))).cuda()


    def encoder(self, mask=None):#将多模态内容特征编码为统一的嵌入表示
        feature = torch.tensor([]).cuda()

        if self.v_feat is not None:
            feature = torch.cat((feature, self.v_feat), dim=1)

        if self.a_feat is not None:
            feature = torch.cat((feature, self.a_feat), dim=1)

        if self.t_feat is not None:
            if self.is_word:
                t_feat = F.normalize(
                    scatter(self.t_feat[self.word_tensor[1]], self.word_tensor[0], dim=0, reduce='mean')).cuda()
                feature = torch.cat((feature, t_feat), dim=1)
            else:
                feature = torch.cat((feature, self.t_feat), dim=1)


        feature = F.leaky_relu(self.encoder_layer1(feature))
        feature = self.encoder_layer2(feature)
        return feature


    def loss(self, user_tensor, item_tensor, graph, content_features):
        """
        Compute multi-task loss: L_UI + L_RE + L_reg

        Args:
            user_tensor: [batch_size, 1+num_neg] User IDs for BPR
            item_tensor: [batch_size, 1+num_neg] Item IDs for BPR
            graph: Normalized sparse adjacency matrix for LightGCN
            content_features: Content feature tensor [num_items, feat_dim]

        Returns:
            total_loss: Combined loss
            loss_ui: U-I recommendation loss
            loss_re: R-E loss
            reg_loss: Regularization loss
        """
        batch_size = user_tensor.size(0)

        # --- 1. LightGCN Encoding ---
        all_emb_list = self.lightgcn(graph)
        Z_collab = self.lightgcn.get_final_embeddings(all_emb_list)

        # --- 2. L_UI (U-I Recommendation Loss) ---
        # Extract embeddings for BPR
        u_ids = user_tensor[:, 0]  # [batch_size]
        u_emb_bpr = Z_collab[u_ids]  # [batch_size, dim]
        i_emb_bpr_all = Z_collab[item_tensor]  # [batch_size, 1+num_neg, dim]

        # Compute scores
        scores = torch.bmm(
            i_emb_bpr_all,  # [batch_size, 1+num_neg, dim]
            u_emb_bpr.unsqueeze(-1)  # [batch_size, dim, 1]
        ).squeeze(-1)  # [batch_size, 1+num_neg]

        scores = torch.exp(scores / self.temperature_ui)

        pos_score = scores[:, 0]  # [batch_size]
        all_score = scores.sum(dim=1)  # [batch_size]

        # InfoNCE-style loss
        loss_ui = -torch.log(pos_score / all_score).mean()

        # --- 3. L_RE (Structured R-E Loss) ---
        # Use only positive items to save memory
        items_for_re = item_tensor[:, 0].unique()  # Only positive items

        # Get content embeddings via encoder (must be encoded to dim_E for comparison with collaborative embeddings)
        full_features = self.encoder()
        items_for_re_offset = items_for_re - self.num_user
        f_batch = full_features[items_for_re_offset]

        # Get collaborative embeddings
        z_batch = Z_collab[items_for_re]

        # Normalize
        f_batch_norm = F.normalize(f_batch, p=2, dim=1)
        z_batch_norm = F.normalize(z_batch, p=2, dim=1)

        # Compute structural positive mask
        batch_size_re = z_batch.size(0)

        if batch_size_re <= 512:
            # Small batch: compute directly
            sim_zz = torch.matmul(z_batch_norm, z_batch_norm.T)
            pos_mask = (sim_zz > self.structural_threshold).float()
            pos_mask.fill_diagonal_(1)
        else:
            # Large batch: use chunked computation
            chunk_size = 256
            pos_mask = torch.zeros(batch_size_re, batch_size_re).cuda()

            for i in range(0, batch_size_re, chunk_size):
                end_i = min(i + chunk_size, batch_size_re)
                sim_chunk = torch.mm(z_batch_norm[i:end_i], z_batch_norm.t())
                pos_mask[i:end_i] = (sim_chunk > self.structural_threshold).float()

            pos_mask.fill_diagonal_(1)

        # Compute L_RE
        loss_re = self.lambda_re * calc_infonce_with_mask(
            f_batch_norm, z_batch_norm, pos_mask, self.temperature_re
        )

        # --- 4. Regularization Loss ---
        # L2 regularization on LightGCN embeddings and encoder parameters
        reg_loss = self.reg_weight * self.lightgcn.embedding.weight.norm(2).pow(2)

        # Add encoder regularization
        for param in self.encoder_layer1.parameters():
            reg_loss += self.reg_weight * param.norm(2).pow(2)
        for param in self.encoder_layer2.parameters():
            reg_loss += self.reg_weight * param.norm(2).pow(2)

        reg_loss = reg_loss / batch_size

        # --- 5. Total Loss ---
        total_loss = loss_ui + loss_re + reg_loss

        return total_loss, loss_ui, loss_re, reg_loss
