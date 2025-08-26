import torch
import os
import logging
import time
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected
import matplotlib.pyplot as plt
from utils import plot_distribution_before_normalization

def normalized_power_simple_min_max(y):
    """
    简单的Min-Max归一化到指定范围
    
    Args:
        y (torch.Tensor): 原始数据
        lower_bound (float): 归一化后的最小值，默认为 0.2
        upper_bound (float): 归一化后的最大值，默认为 0.8
        
    Returns:
        torch.Tensor: 归一化后的数据
    """
    # 直接使用Min-Max归一化
    min_val = y.min()
    max_val = y.max()
    
    # 归一化到[0, 1]
    normalized_y = (y - min_val) / (max_val - min_val)
    
    # 映射到指定范围[lower_bound, upper_bound]

    
    return normalized_y
# def normalized_power_centered_min_max(y, lower_bound=0.2, upper_bound=0.8):
#     """
#     将数据归一化到 [lower_bound, upper_bound] 范围内，
#     并确保大部分数据集中在中间区域。
    
#     Args:
#         y (torch.Tensor): 原始数据。
#         lower_bound (float): 归一化后的最小值，默认为 0.2。
#         upper_bound (float): 归一化后的最大值，默认为 0.8。
        
#     Returns:
#         torch.Tensor: 归一化后的数据。
#     """
#     # 计算均值和标准差
#     mean_y = y.mean()
#     std_y = y.std()

#     # 标准化
#     normalized_y = (y - mean_y) / std_y
    
#     # 将标准化后的数据缩放到 [0, 1]
#     normalized_y = (normalized_y - normalized_y.min()) / (normalized_y.max() - normalized_y.min())
    
#     # 将数据映射到 [lower_bound, upper_bound] 范围
#     stretched_y = normalized_y * (upper_bound - lower_bound) + lower_bound

#     return stretched_y


class SealSramDataset(InMemoryDataset):
    def __init__(
        self,
        name,
        root,
        args=None,  # 添加args参数
        neg_edge_ratio=1.0,
        to_undirected=True,
        task_level='node',
        task_type='classification',  # 'classification' 或 'regression'
        transform=None, 
        pre_transform=None
    ) -> None:
        self.args = args  # 保存args
        self.name = 'sram'
        self.task_type = task_type
        
        # 分类任务数据集
        self.classification_datasets = [
            # 示例：'capacitance_classification_graph' （如果有分类任务）
        ]
        
        # 回归任务数据集（全部归到此处）
        self.regression_datasets = [
            'integrated_position_prediction_graph',            
            'integrated_power_density_prediction_graph',
            'integrated_route_with_global_features',
            'integrated_floorplan_area_prediction_graph'
        ]
        
        # 解析数据集名称并根据任务类型过滤
        if '+' in name:
            all_names = name.split('+')
        else:
            all_names = [name]
        print(f"name:{name}")
        
        # 根据任务类型过滤数据集
        if task_type == 'classification':
            self.names = [n for n in all_names if n in self.classification_datasets]
        elif task_type == 'regression':
            self.names = [n for n in all_names if n in self.regression_datasets]
        else:
            raise ValueError(f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'")
        
        # 如果没有对应任务类型的数据集，设置为空
        if not self.names:
            print(f"No {task_type} datasets found in {all_names}")
            self.empty_dataset = True
            return
        else:
            self.empty_dataset = False
            print(f"SealSramDataset ({task_type}) includes {self.names} circuits")
        
        self.folder = os.path.join(root, self.name)
        self.neg_edge_ratio = neg_edge_ratio
        self.to_undirected = to_undirected
        self.data_lengths = {}
        self.data_offsets = {}
        self.task_level = task_level
    
        # 统一10维
        self.max_net_node_feat = torch.ones((1, 10))
        self.max_dev_node_feat = torch.ones((1, 10))
        
        super().__init__(self.folder, transform, pre_transform)
        
        # 如果是空数据集，直接返回
        if self.empty_dataset:
            return
        
        # 加载所有数据
        data_list = []
        for i, name in enumerate(self.names):
            if os.path.exists(self.processed_paths[i]):
                try:
                    loaded_data, loaded_slices = torch.load(self.processed_paths[i], weights_only=False)
                except Exception:
                    # 尝试其他方法
                    data_list.append(loaded_data)
                
                self.data_offsets[name] = len(data_list)
                if self.task_level == 'node':
                    self.data_lengths[name] = loaded_data.y.size(0) if hasattr(loaded_data, 'y') else loaded_data.num_nodes
                elif self.task_level == 'edge':
                    self.data_lengths[name] = loaded_data.edge_label.size(0) if hasattr(loaded_data, 'edge_label') else loaded_data.num_edges

                data_list.append(loaded_data)
        
        if data_list:
            self.data, self.slices = self.collate(data_list)

    def sram_graph_load(self, name, raw_path):
        logging.info(f"raw_path: {raw_path}")
        g = torch.load(raw_path, weights_only=False, map_location='cpu')
        print(f"g:{g}")
        print(f"Loaded data type: {type(g)}")
        if isinstance(g, tuple):
            print(f"Tuple contents: {g}")
            g = g[0]
        if isinstance(g, list):
            g = g[0]
        
        g.name = name
        
        if hasattr(g, 'node_type'):
            g.x = g.node_type.view(-1, 1)
            
            
        plot_distribution_before_normalization(g.y, self.args.dataset)
        
        
        # 拼接global_features到节点特征
        if self.task_level == 'node':
            if hasattr(g, 'global_features') and hasattr(g, 'graph_id'):
                # global_features: [num_subgraphs, 4]
                # graph_id: [num_nodes] - 每个节点对应的子图ID
                # 根据graph_id索引对应的global_features
                node_global_features = g.global_features[g.graph_id]  # [num_nodes, 4]

                # 将global_features拼接到节点特征x后面
                g.x = torch.cat([g.x, node_global_features], dim=1)  # [num_nodes, 10+4=14]
                if not hasattr(g, 'y') or g.y is None:
                    g.y = torch.zeros(g.num_nodes, 1)
                elif self.args.dataset == 'integrated_power_density_prediction_graph':
                    
                    g.y = normalized_power_simple_min_max(g.y)
                    g.x[:,4] = g.x[:,4] / g.x[:,4].max()
                    g.x[:,6] = g.x[:,6] / g.x[:,6].max()
                    # 将第0列和第1列设置为0
                    g.x[:,0] = 0
                    g.x[:,1] = 0
                    g.edge_attr[:,0] = 0  # 第1列（索引0）
                    g.edge_attr[:,2] = 0  # 第3列（索引2）
                elif self.args.dataset == 'integrated_position_prediction_graph':
                    g.y = g.y/255
                    g.x[:,4] = g.x[:,4] / g.x[:,4].max()
                    g.x[:,6] = g.x[:,6] / g.x[:,6].max()
                    # 将第0列和第1列设置为0
                    g.x[:,0] = 0
                    g.x[:,1] = 0
                    g.edge_attr[:,0] = 0  # 第1列（索引0）
                    g.edge_attr[:,2] = 0  # 第3列（索引2）
                # elif self.args.dataset == 'integrated_route_with_global_features':
                # elif self.args.dataset == 'integrated_floorplan_area_prediction_graph':
        elif self.task_level == 'edge':
            if not hasattr(g, 'y') or g.y is None:
                g.y = torch.zeros(g.num_nodes, 1)
            #  elif self.args.dataset == 'integrated_route_with_global_features':            
            # if hasattr(g, 'edge_index'):
            #     g.edge_label_index = g.edge_index
            #     g.edge_label = torch.zeros(g.num_edges, 1)
        
        if self.to_undirected and hasattr(g, 'edge_index'):
            if hasattr(g, 'edge_type'):
                g.edge_index, g.edge_type = to_undirected(g.edge_index, g.edge_type, g.num_nodes, reduce='mean')
            else:
                g.edge_index = to_undirected(g.edge_index, num_nodes=g.num_nodes)
        
        return g

    def single_g_process(self, idx: int):
        if self.empty_dataset:
            return 0
            
        graph = self.sram_graph_load(self.names[idx], self.raw_paths[idx])
        
        if self.task_level == 'node' or self.task_level == 'edge':
            torch.save((graph, None), self.processed_paths[idx])
            return graph.y.size(0)
        # elif self.task_level == 'edge':
        #     torch.save((graph, None), self.processed_paths[idx])
        #     return graph.edge_label.size(0)

    def process(self):
        if self.empty_dataset:
            return
            
        for i, name in enumerate(self.names):
            if os.path.exists(self.processed_paths[i]):
                continue
            self.single_g_process(i)

    @property
    def raw_file_names(self):
        if self.empty_dataset:
            return []
            
        raw_file_names = []
        for name in self.names:
            raw_file_names.append(name+'.pt')
        return raw_file_names
    
    @property
    def processed_dir(self) -> str:
        if self.task_level == 'edge':
            base_dir = 'processed_for_edges'
        elif self.task_level == 'node':
            base_dir = 'processed_for_nodes'
        else:
            raise ValueError(f"No defination of task {self.task_level}!")
        return os.path.join(self.root, f"{base_dir}_{self.task_type}")

    @property
    def processed_file_names(self):
        if self.empty_dataset:
            return []
            
        processed_names = []
        for i, name in enumerate(self.names):
            if self.neg_edge_ratio < 1.0:
                name += f"_nr{self.neg_edge_ratio:.1f}"
            processed_names.append(name+"_processed.pt")
        return processed_names


def performat_SramDataset(
    dataset_dir,
    name,
    args=None,  # 允许 args 为 None
    neg_edge_ratio=1.0,
    to_undirected=True,
    task_level="node",
):
    start = time.perf_counter()

    try:
        dataset = SealSramDataset(
            name=name,
            root=dataset_dir,
            args=args,  # 传递args
            neg_edge_ratio=neg_edge_ratio,
            to_undirected=to_undirected,
            task_level=task_level,
            task_type="regression",  # 显式指定回归任务
        )
    except Exception as e:
        raise ValueError(f"Failed to load dataset {name}: {str(e)}")

    elapsed = time.perf_counter() - start
    timestr = time.strftime("%H:%M:%S", time.gmtime(elapsed)) + f"{elapsed:.2f}"[-3:]
    print(f"PID = {os.getpid()}")
    print(f"Building datasets from {name} took {timestr}")
    return None, dataset



