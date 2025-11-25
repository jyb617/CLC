import torch
import torch.nn as nn
from tqdm import tqdm

def train(epoch, length, dataloader, model, graph, content_features, optimizer, batch_size, writer):
    """
    Training loop for CLCRec with LightGCN and R-E loss.

    Args:
        epoch: Current epoch number
        length: Total number of samples
        dataloader: Training data loader
        model: CLCRec model
        graph: Normalized sparse adjacency matrix for LightGCN
        content_features: Content feature tensor
        optimizer: Optimizer
        batch_size: Batch size for progress tracking
        writer: TensorBoard writer (optional)

    Returns:
        loss: Final loss value
        dummy: Placeholder for compatibility (always 0.0)
    """
    model.train()
    print('Now, LightGCN-CLCRec training start ...')

    sum_loss = 0.0
    sum_loss_ui = 0.0
    sum_loss_re = 0.0
    sum_reg_loss = 0.0
    step = 0.0

    pbar = tqdm(total=length)
    num_pbar = 0

    for user_tensor, item_tensor in dataloader:
        optimizer.zero_grad()

        # Compute multi-task loss
        loss, loss_ui, loss_re, reg_loss = model.loss(
            user_tensor.cuda(),
            item_tensor.cuda(),
            graph,
            content_features
        )

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Accumulate losses
        sum_loss += loss.cpu().item()
        sum_loss_ui += loss_ui.cpu().item()
        sum_loss_re += loss_re.cpu().item()
        sum_reg_loss += reg_loss.cpu().item()

        pbar.update(batch_size)
        num_pbar += batch_size
        step += 1.0

    pbar.close()

    print('----------------- loss value:{}  UI_loss:{} RE_loss:{} reg_loss:{} --------------'
        .format(sum_loss/step, sum_loss_ui/step, sum_loss_re/step, sum_reg_loss/step))

    # if writer is not None:
    #     writer.add_scalar('Loss/total', sum_loss/step, epoch)
    #     writer.add_scalar('Loss/ui', sum_loss_ui/step, epoch)
    #     writer.add_scalar('Loss/re', sum_loss_re/step, epoch)
    #     writer.add_scalar('Loss/reg', sum_reg_loss/step, epoch)

    return loss, 0.0
