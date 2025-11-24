import torch
import torch.nn.functional as F


def calc_infonce_loss_batch(z1, z2, temperature):
    """
    Calculate InfoNCE loss for a batch of nodes (standard contrastive learning).
    Each node in z1 is paired with its corresponding node in z2 as positive pair.
    All other nodes in the batch are treated as negatives.

    Args:
        z1: Embeddings from view 1, shape [batch_size, dim]
        z2: Embeddings from view 2, shape [batch_size, dim]
        temperature: Temperature parameter for scaling

    Returns:
        InfoNCE loss (scalar)
    """
    batch_size = z1.size(0)

    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Compute similarity matrix: [batch_size, batch_size]
    sim_matrix = torch.mm(z1, z2.t()) / temperature

    # Positive scores are on the diagonal
    pos_scores = torch.diag(sim_matrix)

    # Compute log-sum-exp over all scores for each row
    loss = -pos_scores + torch.logsumexp(sim_matrix, dim=1)

    return loss.mean()


def calc_infonce_with_mask(f_batch, z_batch, pos_mask, temperature):
    """
    Calculate InfoNCE loss with custom positive mask (for R-E loss with structural neighbors).

    For each content embedding f_i, we want it to be close to:
    - Its own collaborative embedding z_i (always positive)
    - Structural neighbors z_j where pos_mask[i,j] = 1

    Args:
        f_batch: Content embeddings, shape [batch_size, dim]
        z_batch: Collaborative embeddings, shape [batch_size, dim]
        pos_mask: Binary mask indicating positive pairs, shape [batch_size, batch_size]
                  pos_mask[i,j] = 1 means (f_i, z_j) is a positive pair
        temperature: Temperature parameter for scaling

    Returns:
        InfoNCE loss (scalar)
    """
    batch_size = f_batch.size(0)

    # Normalize embeddings
    f_batch = F.normalize(f_batch, dim=1)
    z_batch = F.normalize(z_batch, dim=1)

    # Compute similarity matrix: [batch_size, batch_size]
    # sim[i,j] = similarity between f_i and z_j
    sim_matrix = torch.mm(f_batch, z_batch.t()) / temperature

    # Apply exponential
    exp_sim = torch.exp(sim_matrix)

    # Compute positive scores: sum of exp(sim) over all positives for each f_i
    pos_scores = (exp_sim * pos_mask).sum(dim=1)

    # Compute all scores: sum of exp(sim) over all samples
    all_scores = exp_sim.sum(dim=1)

    # InfoNCE loss: -log(pos_scores / all_scores)
    loss = -torch.log(pos_scores / all_scores)

    return loss.mean()


def build_graph(train_data, num_user, num_item):
    """
    Build normalized sparse adjacency matrix for LightGCN.

    Args:
        train_data: Training data, shape [num_edges, 2], each row is (user_id, item_id)
        num_user: Number of users
        num_item: Number of items

    Returns:
        Normalized sparse COO tensor graph
    """
    # Create edge list
    edge_user = torch.LongTensor(train_data[:, 0])
    edge_item = torch.LongTensor(train_data[:, 1])

    # Build bipartite graph edge indices
    edge_index = torch.stack([
        torch.cat([edge_user, edge_item]),
        torch.cat([edge_item, edge_user])
    ], dim=0)

    # Symmetric normalization
    num_nodes = num_user + num_item
    degree = torch.zeros(num_nodes)
    degree.index_add_(0, edge_index[0], torch.ones(edge_index.size(1)))

    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0

    edge_weight = degree_inv_sqrt[edge_index[0]] * degree_inv_sqrt[edge_index[1]]

    # Create sparse COO tensor
    graph = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        size=(num_nodes, num_nodes)
    )

    return graph
