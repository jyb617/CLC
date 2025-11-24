import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import calc_infonce_loss_batch, calc_infonce_with_mask


def train(epoch, length, dataloader, model, graph, optimizer, args):
    """
    Multi-task training loop integrating GCL, R-E, and U-I losses.

    Args:
        epoch: Current epoch number
        length: Total number of samples in dataset
        dataloader: Training data loader (provides BPR batches)
        model: GCL_CLCRec model
        graph: Original sparse adjacency matrix
        optimizer: Optimizer
        args: Arguments containing hyperparameters

    Returns:
        loss: Final loss value
        sum_mat: Placeholder for compatibility
    """
    model.train()
    print('Now, GCL-CLCRec training start ...')

    pbar = tqdm(total=length)
    sum_loss = 0.0
    sum_loss_ui = 0.0
    sum_loss_re = 0.0
    sum_loss_gcl = 0.0
    sum_reg_loss = 0.0
    step = 0.0

    for user_tensor, item_tensor in dataloader:
        optimizer.zero_grad()

        # Original BPR batch: [batch_size, 1+num_neg]
        users_bpr = user_tensor.cuda()  # Shape: [batch_size, 1+num_neg]
        items_bpr = item_tensor.cuda()  # Shape: [batch_size, 1+num_neg]

        # --- Step 1: Node Sampling ---
        # Extract unique users and items from the batch
        users_unique = users_bpr[:, 0].unique()  # Unique users in this batch
        items_unique = items_bpr.view(-1).unique()  # All unique items (pos + neg)

        # --- Step 2: GCL Loss (L_GCL) ---

        # 2a. Graph augmentation
        graph_aug1 = model.graph_augment(graph, args.edge_dropout_rate)
        graph_aug2 = model.graph_augment(graph, args.edge_dropout_rate)

        # 2b. GCL encoding (two forward passes)
        all_emb_v1_list = model.lightgcn(graph_aug1)
        all_emb_v2_list = model.lightgcn(graph_aug2)

        # 2c. Layer combination (get final embeddings)
        Z_view1 = model.lightgcn.get_final_embeddings(all_emb_v1_list)
        Z_view2 = model.lightgcn.get_final_embeddings(all_emb_v2_list)

        # 2d. Extract embeddings for current batch nodes
        z1_u = Z_view1[users_unique]
        z1_i = Z_view1[items_unique]
        z2_u = Z_view2[users_unique]
        z2_i = Z_view2[items_unique]

        # 2e. Compute GCL loss
        loss_gcl_u = calc_infonce_loss_batch(z1_u, z2_u, args.temperature_gcl)
        loss_gcl_i = calc_infonce_loss_batch(z1_i, z2_i, args.temperature_gcl)
        loss_gcl = args.lambda_gcl * (loss_gcl_u + loss_gcl_i)

        # --- Step 3: Structured R-E Loss (L_RE - Method 1) ---

        # 3a. Reuse View1 as main collaborative embeddings
        Z_collab = Z_view1

        # 3b. Get content embeddings for items in batch
        # Note: items_unique includes user+item offset, so we need to subtract num_user
        items_unique_offset = items_unique - model.num_user
        f_batch = model.feature_encoder()  # Get all content embeddings
        f_batch = f_batch[items_unique_offset]  # Select batch items

        # 3c. Get collaborative embeddings for items in batch
        z_batch = Z_collab[items_unique]

        # 3d. Compute structural positive mask (Method 1)
        # Cosine similarity between collaborative embeddings
        sim_zz = F.cosine_similarity(z_batch.unsqueeze(1), z_batch.unsqueeze(0), dim=2)
        pos_mask = (sim_zz > args.structural_threshold).float()
        pos_mask.fill_diagonal_(1)  # Ensure (f_i, z_i) is always positive

        # 3e. Compute R-E loss
        loss_re = args.lambda_re * calc_infonce_with_mask(
            f_batch, z_batch, pos_mask, args.temperature_re
        )

        # --- Step 4: U-I Recommendation Loss (L_UI) ---

        # 4a. Extract user and item embeddings for BPR
        # users_bpr shape: [batch_size, 1+num_neg] but all columns are same user
        # items_bpr shape: [batch_size, 1+num_neg], first column is pos, rest are neg
        batch_size = users_bpr.size(0)
        num_neg = items_bpr.size(1) - 1

        # Get unique user IDs for this batch (one per row)
        u_ids = users_bpr[:, 0]  # Shape: [batch_size]
        u_emb_bpr = Z_collab[u_ids]  # Shape: [batch_size, dim]

        # Get all item embeddings (pos + neg)
        i_ids = items_bpr  # Shape: [batch_size, 1+num_neg]
        i_emb_bpr_all = Z_collab[i_ids]  # Shape: [batch_size, 1+num_neg, dim]

        # 4b. Compute BPR scores
        # Compute dot product: u_emb @ i_emb^T for each sample
        scores = torch.bmm(
            i_emb_bpr_all,  # [batch_size, 1+num_neg, dim]
            u_emb_bpr.unsqueeze(-1)  # [batch_size, dim, 1]
        ).squeeze(-1)  # [batch_size, 1+num_neg]

        pos_scores = scores[:, 0]  # [batch_size]
        neg_scores = scores[:, 1:]  # [batch_size, num_neg]

        # InfoNCE-style loss (as in original CLCRec)
        all_scores_ui = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        loss_ui = -torch.log(
            torch.exp(pos_scores / args.temp_value) /
            torch.exp(all_scores_ui / args.temp_value).sum(1)
        ).mean()

        # --- Step 5: Total Loss and Backpropagation ---

        # L2 regularization loss
        reg_loss = args.reg_weight * model.lightgcn.embedding.weight.norm(2).pow(2) / float(batch_size)

        # Total loss
        loss = loss_ui + loss_re + loss_gcl + reg_loss

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Accumulate losses
        sum_loss += loss.item()
        sum_loss_ui += loss_ui.item()
        sum_loss_re += loss_re.item()
        sum_loss_gcl += loss_gcl.item()
        sum_reg_loss += reg_loss.item()

        pbar.update(batch_size)
        step += 1.0

    pbar.close()

    # Print epoch statistics
    print(f'Epoch {epoch}: '
          f'Total Loss={sum_loss/step:.4f}, '
          f'UI Loss={sum_loss_ui/step:.4f}, '
          f'RE Loss={sum_loss_re/step:.4f}, '
          f'GCL Loss={sum_loss_gcl/step:.4f}, '
          f'Reg Loss={sum_reg_loss/step:.4f}')

    return loss, 0.0
