from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import no_grad
import numpy as np
from Metric import rank, full_accuracy

def full_ranking(epoch, model, graph, data, user_item_inter, mask_items, is_training, step, topk , prefix, writer=None):
    print(prefix+' start...')
    model.eval()
    with no_grad():
        # --- Evaluation logic: Generate model.result ---
        # 1. Propagate on original graph
        all_embeds_list = model.lightgcn(graph)

        # 2. Get final collaborative embeddings
        final_collab_embeds = model.lightgcn.get_final_embeddings(all_embeds_list)

        # 3. Get final content embeddings
        final_content_embeds = model.feature_encoder()

        # 4. Build result tensor for evaluation
        # Users: use collaborative embeddings
        user_embeds = final_collab_embeds[:model.num_user]
        # Warm items: use collaborative embeddings
        warm_item_embeds = final_collab_embeds[model.num_user : model.num_user + model.num_warm_item]
        # Cold items: use content embeddings
        cold_item_embeds = final_content_embeds[model.num_warm_item:]

        model.result = torch.cat([user_embeds, warm_item_embeds, cold_item_embeds], dim=0)
        # --- End evaluation logic ---

        all_index_of_rank_list = rank(model.num_user, user_item_inter, mask_items, model.result, is_training, step, topk)
        precision, recall, ndcg_score = full_accuracy(data, all_index_of_rank_list, user_item_inter, is_training, topk)

        print('---------------------------------{0}-th Precition:{1:.4f} Recall:{2:.4f} NDCG:{3:.4f}---------------------------------'.format(
            epoch, precision, recall, ndcg_score))
        # if writer is not None:
        #     writer.add_scalar(prefix+'_Precition', precision, epoch)
        #     writer.add_scalar(prefix+'_Recall', recall, epoch)
        #     writer.add_scalar(prefix+'_NDCG', ndcg_score, epoch)

        #     writer.add_histogram(prefix+'_visual_distribution', model.v_rep, epoch)
        #     writer.add_histogram(prefix+'_acoustic_distribution', model.a_rep, epoch)
        #     writer.add_histogram(prefix+'_textual_distribution', model.t_rep, epoch)
            
        #     # writer.add_embedding(model.v_rep)
        #     #writer.add_embedding(model.a_rep)
        #     #writer.add_embedding(model.t_rep)
            
        return [precision, recall, ndcg_score]



