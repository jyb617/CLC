import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class LightGCN(nn.Module):
    """
    LightGCN encoder for collaborative filtering.
    """
    def __init__(self, num_nodes, dim_E, num_layers=3):
        """
        Args:
            num_nodes: Total number of nodes (users + items)
            dim_E: Embedding dimension
            num_layers: Number of GCN layers
        """
        super(LightGCN, self).__init__()
        self.num_nodes = num_nodes
        self.dim_E = dim_E
        self.num_layers = num_layers

        # Initial embedding matrix E^(0)
        self.embedding = nn.Embedding(num_nodes, dim_E)
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, graph):
        """
        Forward propagation through K layers.

        Args:
            graph: Sparse adjacency matrix (torch.sparse_coo_tensor)

        Returns:
            List of embeddings [E^(0), E^(1), ..., E^(K)]
        """
        all_embeddings = [self.embedding.weight]
        current_embedding = self.embedding.weight

        for layer in range(self.num_layers):
            # Sparse matrix multiplication: A @ E^(k)
            current_embedding = torch.sparse.mm(graph, current_embedding)
            all_embeddings.append(current_embedding)

        return all_embeddings

    def get_final_embeddings(self, all_embeddings_list):
        """
        Combine embeddings from all layers with equal weights.

        Args:
            all_embeddings_list: List of embeddings [E^(0), E^(1), ..., E^(K)]

        Returns:
            Final embedding E_final = mean([E^(0), E^(1), ..., E^(K)])
        """
        # Stack and compute mean
        final_embeddings = torch.stack(all_embeddings_list, dim=0).mean(dim=0)
        return final_embeddings


class FeatureEncoder(nn.Module):
    """
    Content feature encoder (reused from original CLCRec encoder logic).
    """
    def __init__(self, v_feat, a_feat, t_feat, dim_E, is_word=False):
        """
        Args:
            v_feat: Visual features
            a_feat: Acoustic features
            t_feat: Textual features
            dim_E: Output embedding dimension
            is_word: Whether textual features are word indices (for TikTok dataset)
        """
        super(FeatureEncoder, self).__init__()
        self.v_feat = v_feat
        self.a_feat = a_feat
        self.t_feat = t_feat
        self.is_word = is_word
        self.dim_E = dim_E

        # Calculate total feature dimension
        self.dim_feat = 0

        if v_feat is not None:
            # Register as buffer to ensure it moves with the model
            # Normalize on CPU first to avoid CUDA compatibility issues
            v_feat_cpu = v_feat.cpu() if v_feat.is_cuda else v_feat
            v_feat_normalized = F.normalize(v_feat_cpu, dim=1)
            self.register_buffer('v_feat_norm', v_feat_normalized)
            self.dim_feat += self.v_feat_norm.size(1)
        else:
            self.v_feat_norm = None

        if a_feat is not None:
            # Normalize on CPU first to avoid CUDA compatibility issues
            a_feat_cpu = a_feat.cpu() if a_feat.is_cuda else a_feat
            a_feat_normalized = F.normalize(a_feat_cpu, dim=1)
            self.register_buffer('a_feat_norm', a_feat_normalized)
            self.dim_feat += self.a_feat_norm.size(1)
        else:
            self.a_feat_norm = None

        if t_feat is not None:
            if is_word:
                # For word-based text features (TikTok)
                self.word_embedding = nn.Parameter(
                    nn.init.xavier_normal_(torch.rand((torch.max(t_feat[1]).item() + 1, 128)))
                )
                self.word_tensor = t_feat
                self.dim_feat += 128
            else:
                # Normalize on CPU first to avoid CUDA compatibility issues
                t_feat_cpu = t_feat.cpu() if t_feat.is_cuda else t_feat
                t_feat_normalized = F.normalize(t_feat_cpu, dim=1)
                self.register_buffer('t_feat_norm', t_feat_normalized)
                self.dim_feat += self.t_feat_norm.size(1)
        else:
            self.t_feat_norm = None

        # MLP encoder layers
        self.encoder_layer1 = nn.Linear(self.dim_feat, 256)
        self.encoder_layer2 = nn.Linear(256, dim_E)

    def forward(self, item_indices=None):
        """
        Encode content features into embeddings.

        Args:
            item_indices: Optional item indices to encode (if None, encode all items)

        Returns:
            Content embeddings of shape [num_items, dim_E]
        """
        # Collect all features in a list
        feature_list = []

        # Concatenate all available features
        if self.v_feat is not None:
            if item_indices is not None:
                feature_list.append(self.v_feat_norm[item_indices])
            else:
                feature_list.append(self.v_feat_norm)

        if self.a_feat is not None:
            if item_indices is not None:
                feature_list.append(self.a_feat_norm[item_indices])
            else:
                feature_list.append(self.a_feat_norm)

        if self.t_feat is not None:
            if self.is_word:
                # Word-based text encoding
                t_feat = F.normalize(
                    scatter(self.word_embedding[self.word_tensor[1]],
                           self.word_tensor[0], dim=0, reduce='mean')
                )
                if item_indices is not None:
                    feature_list.append(t_feat[item_indices])
                else:
                    feature_list.append(t_feat)
            else:
                if item_indices is not None:
                    feature_list.append(self.t_feat_norm[item_indices])
                else:
                    feature_list.append(self.t_feat_norm)

        # Concatenate all features
        feature = torch.cat(feature_list, dim=1)

        # Apply MLP
        feature = F.leaky_relu(self.encoder_layer1(feature))
        feature = self.encoder_layer2(feature)

        return feature


class GCL_CLCRec(nn.Module):
    """
    Multi-task learning framework integrating:
    1. Graph Contrastive Learning (GCL)
    2. Structured R-E loss
    3. U-I recommendation loss
    """
    def __init__(self, num_user, num_item, num_warm_item, v_feat, a_feat, t_feat,
                 dim_E, num_layers, reg_weight, temp_value, is_word=False):
        """
        Args:
            num_user: Number of users
            num_item: Number of items
            num_warm_item: Number of warm-start items
            v_feat: Visual features
            a_feat: Acoustic features
            t_feat: Textual features
            dim_E: Embedding dimension
            num_layers: Number of LightGCN layers
            reg_weight: L2 regularization weight
            temp_value: Temperature for U-I loss
            is_word: Whether textual features are word indices
        """
        super(GCL_CLCRec, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.num_warm_item = num_warm_item
        self.dim_E = dim_E
        self.reg_weight = reg_weight
        self.temp_value = temp_value

        # LightGCN encoder
        num_nodes = num_user + num_item
        self.lightgcn = LightGCN(num_nodes, dim_E, num_layers)

        # Feature encoder
        self.feature_encoder = FeatureEncoder(v_feat, a_feat, t_feat, dim_E, is_word)

        # Result tensor for evaluation (will be updated during evaluation)
        self.result = None

    def graph_augment(self, graph, edge_dropout_rate):
        """
        Graph augmentation via edge dropout.

        Args:
            graph: Original sparse graph (torch.sparse_coo_tensor)
            edge_dropout_rate: Dropout rate for edges

        Returns:
            Augmented sparse graph
        """
        if edge_dropout_rate == 0:
            return graph

        # Get edge indices and values
        indices = graph._indices()
        values = graph._values()
        size = graph.size()

        # Create dropout mask
        num_edges = indices.size(1)
        dropout_mask = torch.rand(num_edges).cuda() > edge_dropout_rate

        # Apply mask
        indices_aug = indices[:, dropout_mask]
        values_aug = values[dropout_mask]

        # Renormalize (important!)
        # Recompute degree after dropout
        degree = torch.zeros(size[0]).cuda()
        degree.index_add_(0, indices_aug[0], torch.ones(indices_aug.size(1)).cuda())

        degree_inv_sqrt = degree.pow(-0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0

        # Recompute edge weights
        values_aug = degree_inv_sqrt[indices_aug[0]] * degree_inv_sqrt[indices_aug[1]]

        # Create augmented graph
        graph_aug = torch.sparse_coo_tensor(
            indices_aug,
            values_aug,
            size=size
        )

        return graph_aug
