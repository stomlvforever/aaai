import copy
import torch
import networkx as nx
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader,NeighborLoader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import pickle
from collections import Counter
from torch_geometric.data import HeteroData
from utils import sample_nodes_by_ratio, plot_true_values_distribution_before_sampling  # 导入新模块中的函数
# NET = 0
# DEV = 1
# PIN = 2

# 移除原来的 sample_nodes_by_ratio 函数，因为已经移到 sampling_utils.py 中


def dataset_sampling(args, dataset):
    """ 
    Sampling subgraphs for each graph in dataset
    Args:
        args (argparse.Namespace): The arguments
        dataset (torch_geometric.data.InMemoryDataset): The dataset
    Return:
        train_loader, val_loader, test_loaders
    """
    # 在采样之前绘制真实值分布图
    plot_true_values_distribution_before_sampling(
        dataset, 
        dataset_name=args.dataset
    )
    # print(f"dataset[0].x[:,0].min:{dataset[0].x[:,0].min()}")
    # print(f"dataset[0].x[:,0].max:{dataset[0].x[:,0].max()}")    
    # print(f"dataset[0].x[:,1].min:{dataset[0].x[:,1].min()}")   
    # print(f"dataset[0].x[:,1].max:{dataset[0].x[:,1].max()}")   
    # # print(f"dataset[0].x:{dataset[0].x}")

    # print(f"dataset[0]:{dataset[0]}")
    # print(f"dataset[0].y.unique:{dataset[0].y.unique()}")
    # print(f"dataset[0].edge_index:{dataset[0].edge_index}")    
    # print(f"dataset[0].edge_attr:{dataset[0].edge_attr}")
    # assert 0
    
    all_graph_indices = np.arange(30)
    np.random.shuffle(all_graph_indices)
    train_graph_indices = all_graph_indices[:15]
    val_graph_indices = all_graph_indices[15:20]
    test_graph_indices = all_graph_indices[20:]
    
    print(f"训练子图索引: {train_graph_indices}")
    print(f"验证子图索引: {val_graph_indices}")
    print(f"测试子图索引: {test_graph_indices}")

    # 获取完整图数据 (假设dataset[0]包含整合的30个子图)
    full_graph = dataset[0]
    
    # 按比例采样训练集、验证集、测试集的节点（比如 20%）
    train_node_ind = sample_nodes_by_ratio(full_graph, ratio=args.sample_ratio)  
    val_node_ind = sample_nodes_by_ratio(full_graph, ratio=args.sample_ratio)  
    test_node_ind = sample_nodes_by_ratio(full_graph, ratio=args.sample_ratio)  

    # 创建数据加载器
    train_loader = NeighborLoader(
        full_graph,
        num_neighbors=args.num_hops * [args.num_neighbors],
        input_nodes=train_node_ind,  # 采样后的训练集节点
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    val_loader = NeighborLoader(
        full_graph,
        num_neighbors=args.num_hops * [args.num_neighbors],
        input_nodes=val_node_ind,  # 采样后的验证集节点
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 修改测试集处理逻辑：为每个测试子图创建独立的采样节点加载器
    # test_loaders = {}
    # for i, test_idx in enumerate(test_graph_indices):
    #     # 创建只包含当前测试子图的临时图数据
    #     test_mask = (full_graph.graph_id == test_idx)
    #     test_subgraph_nodes = torch.nonzero(test_mask).squeeze()
        
    #     # 创建临时子图用于采样
    #     temp_graph = Data(
    #         x=full_graph.x[test_subgraph_nodes],
    #         edge_index=full_graph.edge_index,  # 保持原始边索引
    #         graph_id=torch.full((len(test_subgraph_nodes),), test_idx)
    #     )
        
    #     # 对当前测试子图进行采样，使用不同的种子
    #     sampled_test_nodes = sample_nodes_by_ratio(temp_graph, ratio=args.sample_ratio, seed=50+i)
    #     # 将采样结果映射回原始图的节点索引
    #     actual_test_nodes = test_subgraph_nodes[sampled_test_nodes]

    #     graph_name = dataset.graph_names[test_idx]
    #     test_loaders[graph_name] = NeighborLoader(
    #         full_graph,
    #         num_neighbors=args.num_hops * [args.num_neighbors],
    #         input_nodes=actual_test_nodes,  # 采样后的测试集节点
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         num_workers=args.num_workers,
    #     )
    test_loader = NeighborLoader(
        full_graph,
        num_neighbors=args.num_hops * [args.num_neighbors],
        input_nodes=test_node_ind,  # 直接使用第83行的采样结果
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    print(f"train_loader:{train_loader}, val_loader:{val_loader}, test_loader:{test_loader}")
    
    max_label = None
    return (train_loader, val_loader, test_loader, max_label)