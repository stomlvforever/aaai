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
from utils import  plot_true_values_distribution_before_sampling, get_nodes_from_subgraphs  # 导入新模块中的函数



# 移除原来的 sample_nodes_by_ratio 函数，因为已经移到 utils.py 中


def dataset_sampling(args, dataset):
    """
    对数据集中的每个图进行子图采样，创建训练、验证和测试数据加载器
    
    该函数实现了图神经网络训练中的数据采样策略，将完整的图数据分割为训练集、验证集和测试集，
    并为每个数据集创建相应的邻居采样加载器。采样过程包括：
    1. 随机划分30个子图为训练/验证/测试集
    2. 按比例从每个数据集中采样节点
    3. 创建NeighborLoader进行批量加载
    
    Args:
        args (argparse.Namespace): 命令行参数对象，包含以下必需属性：
            - dataset (str): 数据集名称，用于保存分布图
            - sample_ratio (float): 节点采样比例，范围[0, 1]，如0.2表示采样20%的节点
            - num_hops (int): 邻居采样的跳数，决定子图的深度
            - num_neighbors (int): 每跳采样的邻居数量
            - batch_size (int): 批处理大小
            - num_workers (int): 数据加载的工作进程数
        dataset (torch_geometric.data.InMemoryDataset): PyTorch Geometric数据集对象
            - 必须包含dataset[0]，即完整的图数据
            - dataset[0]应包含x（节点特征）、edge_index（边索引）、y（标签）、graph_id（子图标识）等属性
    
    Returns:
        tuple: 包含四个元素的元组
            - train_loader (torch_geometric.loader.NeighborLoader): 训练集数据加载器
            - val_loader (torch_geometric.loader.NeighborLoader): 验证集数据加载器  
            - test_loader (torch_geometric.loader.NeighborLoader): 测试集数据加载器
            - max_label (None): 最大标签值，当前实现中返回None
    
    Side Effects:
        - 调用plot_true_values_distribution_before_sampling()绘制并保存数据分布图
        - 打印训练、验证、测试子图的索引信息
        - 打印创建的数据加载器信息
    
    Note:
        - 当前实现固定使用30个子图，前10个作为训练集，第11-15个作为验证集，第16个及以后作为测试集
        - 所有数据集使用相同的采样比例
        - 测试集的高级采样逻辑（按子图分别采样）已被注释，当前使用简化版本
    """
    # 在采样之前绘制真实值分布图，用于数据分析和可视化
    plot_true_values_distribution_before_sampling(
        dataset, 
        dataset_name=args.dataset
    )
    
    # 调试信息（已注释）：用于检查节点特征的数值范围
    # print(f"dataset[0].x[:,0].min:{dataset[0].x[:,0].min()}")
    # print(f"dataset[0].x[:,0].max:{dataset[0].x[:,0].max()}")    
    # print(f"dataset[0].x[:,1].min:{dataset[0].x[:,1].min()}")   
    # print(f"dataset[0].x[:,1].max:{dataset[0].x[:,1].max()}")   
    # print(f"dataset[0].x:{dataset[0].x}")
    # print(f"dataset[0]:{dataset[0]}")
    # print(f"dataset[0].y.unique:{dataset[0].y.unique()}")
    # print(f"dataset[0].edge_index:{dataset[0].edge_index}")    
    # print(f"dataset[0].edge_attr:{dataset[0].edge_attr}")
    # assert 0
    
    # 创建30个子图的索引数组并随机打乱
    all_graph_indices = np.arange(30)
    np.random.shuffle(all_graph_indices)
    
    # 按固定比例划分训练、验证、测试集
    train_graph_indices = all_graph_indices[:10]      # 前10个子图用于训练
    val_graph_indices = all_graph_indices[10:15]      # 第11-15个子图用于验证
    test_graph_indices = all_graph_indices[15:]       # 第16个及以后的子图用于测试
    
    print(f"训练子图索引: {train_graph_indices}")
    print(f"验证子图索引: {val_graph_indices}")
    print(f"测试子图索引: {test_graph_indices}")

    # 获取完整图数据
    full_graph = dataset[0]
    
    # 从指定子图中获取节点，可选择性地进行采样
    train_node_ind = get_nodes_from_subgraphs(full_graph, train_graph_indices, args.sample_ratio)
    val_node_ind = get_nodes_from_subgraphs(full_graph, val_graph_indices, args.sample_ratio)
    test_node_ind = get_nodes_from_subgraphs(full_graph, test_graph_indices, args.sample_ratio)
    
    # print(f"训练节点数量: {len(train_node_ind)}")
    # print(f"验证节点数量: {len(val_node_ind)}")
    # print(f"测试节点数量: {len(test_node_ind)}")
    # assert 0
    # 创建训练集数据加载器
    # NeighborLoader实现邻居采样策略，用于大图的小批量训练
    train_loader = NeighborLoader(
        full_graph,
        num_neighbors=args.num_hops * [args.num_neighbors],  # 每跳的邻居采样数量列表
        input_nodes=train_node_ind,  # 采样后的训练集节点索引
        batch_size=args.batch_size,  # 批处理大小
        shuffle=True,                # 训练时打乱数据顺序
        num_workers=args.num_workers, # 并行加载的工作进程数
    )
    
    # 创建验证集数据加载器
    val_loader = NeighborLoader(
        full_graph,
        num_neighbors=args.num_hops * [args.num_neighbors],
        input_nodes=val_node_ind,    # 采样后的验证集节点索引
        batch_size=args.batch_size,
        shuffle=False,               # 验证时不打乱数据顺序
        num_workers=args.num_workers,
    )

    # 高级测试集处理逻辑（已注释）：为每个测试子图创建独立的采样节点加载器
    # 这种方法可以确保每个测试子图都有独立的采样结果，提供更细粒度的测试控制
    # test_loaders = {}
    # for i, test_idx in enumerate(test_graph_indices):
    #     # 创建只包含当前测试子图的临时图数据
    #     test_mask = (full_graph.graph_id == test_idx)
    #     test_subgraph_nodes = torch.nonzero(test_mask).squeeze()
    #     
    #     # 创建临时子图用于采样
    #     temp_graph = Data(
    #         x=full_graph.x[test_subgraph_nodes],
    #         edge_index=full_graph.edge_index,  # 保持原始边索引
    #         graph_id=torch.full((len(test_subgraph_nodes),), test_idx)
    #     )
    #     
    #     # 对当前测试子图进行采样，使用不同的种子确保可重复性
    #     sampled_test_nodes = sample_nodes_by_ratio(temp_graph, ratio=args.sample_ratio, seed=50+i)
    #     # 将采样结果映射回原始图的节点索引
    #     actual_test_nodes = test_subgraph_nodes[sampled_test_nodes]
    #
    #     graph_name = dataset.graph_names[test_idx]
    #     test_loaders[graph_name] = NeighborLoader(
    #         full_graph,
    #         num_neighbors=args.num_hops * [args.num_neighbors],
    #         input_nodes=actual_test_nodes,  # 采样后的测试集节点
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         num_workers=args.num_workers,
    #     )
    
    # 创建测试集数据加载器（简化版本）
    test_loader = NeighborLoader(
        full_graph,
        num_neighbors=args.num_hops * [args.num_neighbors],
        input_nodes=test_node_ind,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    print(f"train_loader:{train_loader}, val_loader:{val_loader}, test_loader:{test_loader}")
    
    max_label = None
    return (train_loader, val_loader, test_loader, max_label)