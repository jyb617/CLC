import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import random
from Dataset import TrainingDataset, data_load
from model_CLCRec import CLCRec
from torch.utils.data import DataLoader
from Train import train
from Full_rank import full_ranking
from torch.utils.tensorboard import SummaryWriter
###############################248###########################################

def init():#初始化参数
    parser = argparse.ArgumentParser()#ArgumentParser = 参数解析器，用于创建一个命令行接口，让程序能够接收和处理用户从终端传入的参数。
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--data_path', default='movielens', help='Dataset path')
    parser.add_argument('--save_file', default='', help='Filename')

    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default=None, help='Writing weight filename.')
    parser.add_argument('--prefix', default='', help='Prefix of save_file.')

    parser.add_argument('--l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--reg_weight', type=float, default=1e-1, help='Weight decay.')
    parser.add_argument('--model_name', default='SSL', help='Model Name.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--num_neg', type=int, default=512, help='Negative size.')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')

    parser.add_argument('--dim_E', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--topK', type=int, default=10, help='Workers number.')
    parser.add_argument('--step', type=int, default=2000, help='Workers number.')

    # New hyperparameters for LightGCN and R-E losses
    parser.add_argument('--num_layers', type=int, default=3, help='Number of LightGCN layers.')
    parser.add_argument('--lambda_re', type=float, default=0.1, help='Weight for R-E loss.')
    parser.add_argument('--temperature_re', type=float, default=0.1, help='Temperature for R-E loss.')
    parser.add_argument('--temperature_ui', type=float, default=1.0, help='Temperature for U-I loss.')
    parser.add_argument('--structural_threshold', type=float, default=0.8, help='Cosine similarity threshold for structural positives in R-E loss.')

    parser.add_argument('--has_v', default='False', help='Has Visual Features.')
    parser.add_argument('--has_a', default='False', help='Has Acoustic Features.')
    parser.add_argument('--has_t', default='False', help='Has Textual Features.')

    args = parser.parse_args()#parse_args() = 解析参数，读取命令行输入，匹配定义好的参数，返回一个包含所有参数值的对象。
    return args


if __name__ == '__main__':
    args = init()
    #设置随机种子
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)#cpu随机种子
    torch.cuda.manual_seed_all(seed)#gpu随机种子
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    ##########################################################################################################################################
    #设置参数
    data_path = args.data_path
    save_file_name = args.save_file

    learning_rate = args.l_r
    reg_weight = args.reg_weight
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epoch = args.num_epoch
    num_neg = args.num_neg
    topK = args.topK
    prefix = args.prefix
    model_name = args.model_name
    step = args.step
    has_v = True if args.has_v == 'True' else False
    has_a = True if args.has_a == 'True' else False
    has_t = True if args.has_t == 'True' else False

    dim_E = args.dim_E
    is_word = True if data_path == 'tiktok' else False
    writer = SummaryWriter()#日志记录，可视化
    # with open(data_path+'/result/result{0}_{1}.txt'.format(l_r, reg_weight), 'w') as save_file:
    #     save_file.write('---------------------------------lr: {0} \t reg_weight:{1} ---------------------------------\r\n'.format(l_r, reg_weight))
    ##########################################################################################################################################
    print('Data loading ...')
    #加载数据
    num_user, num_item, num_warm_item, train_data, val_data, val_warm_data, val_cold_data, test_data, test_warm_data, test_cold_data, a_feat, t_feat = data_load(data_path)
    
    dir_str = './Data/' + data_path
    user_item_all_dict = np.load(dir_str+'/user_item_dict.npy', allow_pickle=True).item()#item()提取Python对象（dict, list等）
    user_item_train_dict = np.load(dir_str+'/user_item_train_dict.npy', allow_pickle=True).item()

    warm_item = torch.tensor(np.load(dir_str + '/warm_set.npy'))
    cold_item = torch.tensor(np.load(dir_str + '/cold_set.npy'))

    train_dataset = TrainingDataset(num_user, num_item, user_item_all_dict, data_path, train_data, num_neg)#创建PyTorch Dataset对象，封装训练数据和负采样逻辑
    
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)

    print('Data has been loaded.')
    ##########################################################################################################################################
    # Build content features tensor (concatenate all modalities)
    print('Building content features...')
    content_feature_list = []

    if has_a and a_feat is not None:
        content_feature_list.append(F.normalize(a_feat, dim=1))

    if has_t and t_feat is not None:
        if is_word:
            # For word-based features, keep as-is (will be processed in model)
            content_features = None  # Will be handled by model
        else:
            content_feature_list.append(F.normalize(t_feat, dim=1))

    # Concatenate all features
    if content_feature_list:
        content_features = torch.cat(content_feature_list, dim=1).cuda()
    else:
        content_features = None

    print('Content features built.')
    ##########################################################################################################################################
    # Build graph for LightGCN
    print('Building graph...')
    # Create edge list from train_data: (user, item) pairs
    edge_user = torch.LongTensor(train_data[:, 0])
    edge_item = torch.LongTensor(train_data[:, 1])

    # Build adjacency matrix indices (user-item bipartite graph)
    # User nodes: 0 to num_user-1
    # Item nodes: num_user to num_user+num_item-1
    edge_index = torch.stack([
        torch.cat([edge_user, edge_item]),
        torch.cat([edge_item, edge_user])
    ], dim=0)

    # Symmetric normalization: D^(-1/2) A D^(-1/2)
    num_nodes = num_user + num_item
    # Count degree for each node
    degree = torch.zeros(num_nodes)
    degree.index_add_(0, edge_index[0], torch.ones(edge_index.size(1)))

    # Compute D^(-1/2)
    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0

    # Compute edge weights: D^(-1/2)[i] * D^(-1/2)[j]
    edge_weight = degree_inv_sqrt[edge_index[0]] * degree_inv_sqrt[edge_index[1]]

    # Create sparse COO tensor
    graph = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        size=(num_nodes, num_nodes)
    ).cuda()

    print('Graph has been built.')
    ##########################################################################################################################################
    # Create CLCRec model with LightGCN
    model = CLCRec(
        num_user=num_user,
        num_item=num_item,
        num_warm_item=num_warm_item,
        edge_index=train_data,
        reg_weight=reg_weight,
        dim_E=dim_E,
        v_feat=None,  # Visual features (not used in movielens)
        a_feat=a_feat,
        t_feat=t_feat,
        num_neg=num_neg,
        is_word=is_word,
        # New parameters
        num_layers=args.num_layers,
        lambda_re=args.lambda_re,
        temperature_re=args.temperature_re,
        temperature_ui=args.temperature_ui,
        structural_threshold=args.structural_threshold
    ).cuda()
    
    ##########################################################################################################################################
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}])#, 'weight_decay': reg_weight}])
    ##########################################################################################################################################
    max_precision = 0.0
    max_recall = 0.0
    max_NDCG = 0.0
    num_decreases = 0 
    max_val_result = max_val_result_warm = max_val_result_cold = max_test_result = max_test_result_warm = max_test_result_cold = list()
    for epoch in range(num_epoch):
        loss, mat = train(epoch, len(train_dataset), train_dataloader, model, graph, content_features, optimizer, batch_size, writer)

        if torch.isnan(loss):
            print(model.result)
            with open('./Data/'+data_path+'/result_{0}.txt'.format(save_file_name), 'a') as save_file:
                    save_file.write('lr:{0} \t reg_weight:{1} is Nan\r\n'.format( learning_rate, reg_weight))
            break
        torch.cuda.empty_cache()

        # train_precision, train_recall, train_ndcg = full_ranking(epoch, model, graph, content_features, user_item_inter, user_item_inter, True, step, topK, 'Train', writer)
        val_result = full_ranking(epoch, model, graph, content_features, val_data, user_item_train_dict, None, False, step, topK, 'Val/', writer)

        val_result_warm = full_ranking(epoch, model, graph, content_features, val_warm_data, user_item_train_dict, cold_item, False, step, topK, 'Val/warm_', writer)

        val_result_cold = full_ranking(epoch, model, graph, content_features, val_cold_data, user_item_train_dict, warm_item, False, step, topK, 'Val/cold_', writer)

        test_result = full_ranking(epoch, model, graph, content_features, test_data, user_item_train_dict, None, False, step, topK, 'Test/', writer)

        test_result_warm = full_ranking(epoch, model, graph, content_features, test_warm_data, user_item_train_dict, cold_item, False, step, topK, 'Test/warm_', writer)

        test_result_cold = full_ranking(epoch, model, graph, content_features, test_cold_data, user_item_train_dict, warm_item, False, step, topK, 'Test/cold_', writer)


        if val_result[1] > max_recall:#更新最佳参数
            pre_id_embedding = model.lightgcn.embedding.weight
            max_recall = val_result[1]
            max_val_result = val_result
            max_val_result_warm = val_result_warm
            max_val_result_cold = val_result_cold
            max_test_result = test_result
            max_test_result_warm = test_result_warm
            max_test_result_cold = test_result_cold
            num_decreases = 0#早停计数
        else:
            if num_decreases > 5:#早停机制
                with open('./Data/'+data_path+'/result_{0}.txt'.format(save_file_name), 'a') as save_file:
                    save_file.write(str(args))
                    save_file.write('\r\n-----------Val Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------'.format(max_val_result[0], max_val_result[1], max_val_result[2]))
                    save_file.write('\r\n-----------Val Warm Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------'.format(max_val_result_warm[0], max_val_result_warm[1], max_val_result_warm[2]))
                    save_file.write('\r\n-----------Val Cold Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------'.format(max_val_result_cold[0], max_val_result_cold[1], max_val_result_cold[2]))
                    save_file.write('\r\n-----------Test Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------'.format(max_test_result[0], max_test_result[1], max_test_result[2]))
                    save_file.write('\r\n-----------Test Warm Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------'.format(max_test_result_warm[0], max_test_result_warm[1], max_test_result_warm[2]))
                    save_file.write('\r\n-----------Test Cold Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------'.format(max_test_result_cold[0], max_test_result_cold[1], max_test_result_cold[2]))
                break
            else:
                num_decreases += 1