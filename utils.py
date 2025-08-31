import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import r2_score

def get_nodes_from_subgraphs(full_graph, subgraph_indices, sample_ratio=1.0):
    """
    从指定的子图中获取节点索引，可选择性地进行采样
    
    Args:
        full_graph: 完整的图数据
        subgraph_indices: 子图索引列表
        sample_ratio: 采样比例，1.0表示使用所有节点
    
    Returns:
        采样后的节点索引
    """
    all_nodes = []
    
    for subgraph_idx in subgraph_indices:
        # 获取属于当前子图的所有节点
        subgraph_mask = (full_graph.graph_id == subgraph_idx)
        subgraph_nodes = torch.nonzero(subgraph_mask).squeeze()
        
        if sample_ratio < 1.0:
            # 对当前子图的节点进行采样
            num_nodes = len(subgraph_nodes)
            num_sample = max(1, int(num_nodes * sample_ratio))
            
            # 随机采样节点
            perm = torch.randperm(num_nodes)
            sampled_indices = perm[:num_sample]
            sampled_nodes = subgraph_nodes[sampled_indices]
            all_nodes.append(sampled_nodes)
        else:
            # 使用所有节点
            all_nodes.append(subgraph_nodes)
    
    # 合并所有采样的节点
    if all_nodes:
        return torch.cat(all_nodes)
    else:
        return torch.tensor([], dtype=torch.long)


def plot_true_values_distribution_before_sampling(dataset, dataset_name=None, save_dir="distribution_plots"):
    """
    在采样之前绘制真实值的分布直方图，包括频率分布、密度分布、累积分布和箱线图
    
    Args:
        dataset: 数据集对象，必须支持索引访问dataset[0]，且包含y属性
        dataset_name (str, optional): 数据集名称（如 integrated_power_density_prediction_graph），用于图表标题和文件命名
        save_dir (str): 保存图片的目录，默认为"distribution_plots"
    
    Returns:
        str: 保存的图片文件路径
    
    Side Effects:
        - 创建保存目录（如果不存在）
        - 保存包含4个子图的分布图像文件
        - 打印详细的统计信息到控制台
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取完整图数据
    full_graph = dataset[0]
    
    # 获取真实值
    true_values = full_graph.y.cpu().numpy()
    
    # 设置数据集名称显示
    dataset_display = f" - {dataset_name}" if dataset_name else ""
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制直方图
    plt.subplot(2, 2, 1)
    plt.hist(true_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'True Values Distribution {dataset_display}')
    plt.xlabel('True Values')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 绘制密度图
    plt.subplot(2, 2, 2)
    plt.hist(true_values, bins=50, alpha=0.7, color='green', density=True, edgecolor='black')
    plt.title(f'True Values Density Distribution{dataset_display}')
    plt.xlabel('True Values')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # 绘制累积分布
    plt.subplot(2, 2, 3)
    sorted_values = np.sort(true_values)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    plt.plot(sorted_values, cumulative, color='red', linewidth=2)
    plt.title(f'Cumulative Distribution{dataset_display}')
    plt.xlabel('True Values')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, alpha=0.3)
    
    # 绘制箱线图
    plt.subplot(2, 2, 4)
    plt.boxplot(true_values, vert=True)
    plt.title(f'True Values Box Plot{dataset_display}')
    plt.ylabel('True Values')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片（文件名包含数据集名称）
    filename = f'{dataset_name}_true_values_distribution_before_sampling.png' if dataset_name else 'true_values_distribution_before_sampling.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印统计信息
    print(f"\n=== 真实值分布统计信息 (采样前){dataset_display} ===")
    print(f"总样本数: {len(true_values)}")
    print(f"最小值: {true_values.min():.4f}")
    print(f"最大值: {true_values.max():.4f}")
    print(f"均值: {true_values.mean():.4f}")
    print(f"标准差: {true_values.std():.4f}")
    print(f"中位数: {np.median(true_values):.4f}")
    print(f"25%分位数: {np.percentile(true_values, 25):.4f}")
    print(f"75%分位数: {np.percentile(true_values, 75):.4f}")
    print(f"分布图已保存至: {save_path}")
    print(f"================================\n")
    
    return save_path

def plot_pred_vs_true_scatter(pred_values, true_values, epoch, split_name, save_dir=None, device=None):
    """
    绘制真实值vs预测值的散点图，包括R²和MAE指标，支持GPU加速计算
    
    Args:
        pred_values (torch.Tensor or array-like): 预测值数组，可以是torch.Tensor或numpy数组
        true_values (torch.Tensor or array-like): 真实值数组，可以是torch.Tensor或numpy数组
        epoch (int): 当前epoch数，用于文件命名和图表标题
        split_name (str): 数据集名称 (train/val/test)，用于文件命名和图表标题
        save_dir (str, optional): 保存目录，如果为None则自动生成基于PID的目录
        device (torch.device, optional): 计算设备，如果为None则自动检测
    
    Returns:
        None
    
    Side Effects:
        - 创建保存目录（如果不存在）
        - 保存散点图文件
        - 打印保存路径信息到控制台
    """
    # 动态生成保存目录（基于进程ID）
    if save_dir is None:
        pid = os.getpid()
        save_dir = f"scatter_plots_pid_{pid}"
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 自动检测设备
    if device is None:
        if torch.is_tensor(pred_values) and pred_values.is_cuda:
            device = pred_values.device
        elif torch.is_tensor(true_values) and true_values.is_cuda:
            device = true_values.device
        else:
            device = torch.device('cpu')
    
    # 在GPU上进行计算
    if torch.is_tensor(pred_values):
        pred_values = pred_values.to(device).flatten()
    else:
        pred_values = torch.tensor(pred_values, device=device).flatten()
        
    if torch.is_tensor(true_values):
        true_values = true_values.to(device).flatten()
    else:
        true_values = torch.tensor(true_values, device=device).flatten()
    
    # 在GPU上计算统计量
    min_val = torch.min(torch.min(pred_values), torch.min(true_values))
    max_val = torch.max(torch.max(pred_values), torch.max(true_values))
    
    # 计算R²和MAE（在GPU上）
    mae = torch.mean(torch.abs(pred_values - true_values))
    
    # 计算R² (在GPU上)
    true_mean = torch.mean(true_values)
    ss_tot = torch.sum((true_values - true_mean) ** 2)
    ss_res = torch.sum((true_values - pred_values) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # 只在绘图时转换到CPU
    pred_cpu = pred_values.cpu().numpy()
    true_cpu = true_values.cpu().numpy()
    min_val_cpu = min_val.cpu().item()
    max_val_cpu = max_val.cpu().item()
    r2_cpu = r2.cpu().item()
    mae_cpu = mae.cpu().item()
    
    # 创建散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(true_cpu, pred_cpu, alpha=0.6, s=20)
    plt.plot([min_val_cpu, max_val_cpu], [min_val_cpu, max_val_cpu], 'r--', linewidth=2, label='Perfect Prediction')
    
    # 设置图表属性
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'True vs Pred - Epoch {epoch} ({split_name})\nR² = {r2_cpu:.4f}, MAE = {mae_cpu:.4f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    filename = f"{save_dir}/scatter_epoch_{epoch}_{split_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"散点图已保存: {filename} (PID: {os.getpid()})")

# def collect_predictions_for_plot(args, loader, model, device):
#     """
#     收集所有预测值和真实值用于绘图（保持在GPU上）
#     """
#     model.eval()
#     all_preds = []
#     all_trues = []
    
#     with torch.no_grad():
#         for batch in loader:
#             pred, _, label_true = model(batch.to(device))
#             if args.task == 'regression':
#                 # 确保pred是[N, 1]形状
#                 if pred.ndim == 1:
#                     pred = pred.view(-1, 1)
#                 elif pred.size(1) != 1:
#                     pred = pred.view(-1, 1)
                
#                 # 处理true张量的维度
#                 true = label_true.float()
#                 if true.ndim == 1:
#                     true = true.view(-1, 1)
#                 elif true.size(1) != 1:
#                     true = true[:, 0].view(-1, 1)
                
#                 pred = pred.float()
                
#                 # pred = pred.squeeze()
#                 # # 添加调试信息
#                 # print(f"散点图收集 - pred范围: {pred.min():.4f} ~ {pred.max():.4f}")                
#                 # 保持在GPU上，不转换到CPU
#                 all_preds.append(pred)
#                 all_trues.append(true)
    
#     # 合并所有批次的数据（仍在GPU上）
#     all_preds = torch.cat(all_preds, dim=0)
#     all_trues = torch.cat(all_trues, dim=0)
    
#     return all_preds, all_trues

def plot_distribution_before_normalization(data, dataset_name, save_dir="distribution_plots", device=None):
    """
    绘制归一化前的数据分布图，支持GPU计算统计量
    
    Args:
        data (torch.Tensor or array-like): 需要绘制分布的数据，可以是torch.Tensor或numpy数组
        dataset_name (str): 数据集名称，用于图表标题和文件命名
        save_dir (str): 保存目录，默认为"distribution_plots"
        device (torch.device, optional): 计算设备，如果为None则自动检测
    
    Returns:
        None
    
    Side Effects:
        - 创建保存目录（如果不存在）
        - 保存分布图文件
        - 显示图表
        - 打印详细的统计信息到控制台
    """
    # 自动检测设备
    if device is None and torch.is_tensor(data):
        device = data.device
    
    # 在GPU上计算统计量
    if torch.is_tensor(data):
        data_tensor = data.to(device) if device else data
        min_val = torch.min(data_tensor).item()
        max_val = torch.max(data_tensor).item()
        mean_val = torch.mean(data_tensor.float()).item()
        std_val = torch.std(data_tensor.float()).item()
        median_val = torch.quantile(data_tensor.flatten().float(), 0.5).item()
        
        # 只在绘图时转换到CPU
        data_np = data_tensor.cpu().numpy()
    else:
        data_np = np.array(data)
        min_val = data_np.min()
        max_val = data_np.max()
        mean_val = data_np.mean()
        std_val = data_np.std()
        median_val = np.median(data_np.flatten())
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.hist(data_np, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'True Values Distribution - {dataset_name}')
    plt.xlabel('True Values')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图像
    filename = f'before_normalization_{dataset_name}.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # 打印统计信息
    print(f"\n=== 归一化前数据分布统计 - {dataset_name} ===")
    print(f"最小值: {min_val:.6f}")
    print(f"最大值: {max_val:.6f}")
    print(f"均值: {mean_val:.6f}")
    print(f"标准差: {std_val:.6f}")
    print(f"中位数: {median_val:.6f}")
    print(f"图像已保存至: {save_path}")
    print(f"==============================\n")
    
def convert_position_to_classification_labels(y_continuous, num_classes=10):
    """
    将position任务的连续值转换为分类标签，将[0,1]范围等分为指定数量的类别
    
    Args:
        y_continuous (torch.Tensor): 连续值标签，范围应在[0, 1]之间，形状为[N] 或 [N, 1]
        num_classes (int): 分类数量，默认为10，表示将[0,1]范围分成10个等间隔的类别
    
    Returns:
        torch.Tensor: 分类标签，范围[0, num_classes-1]，数据类型为long，形状与输入相同
    
    Example:
        >>> y_continuous = torch.tensor([0.05, 0.15, 0.95, 1.0])
        >>> labels = convert_position_to_classification_labels(y_continuous, num_classes=10)
        >>> print(labels)  # tensor([0, 1, 9, 9])
    """
    # 确保值在[0, 1]范围内
    y_continuous = torch.clamp(y_continuous, 0.0, 1.0)
    
    # 将[0, 1]范围分成num_classes个区间
    # 0-0.1 -> 0, 0.1-0.2 -> 1, ..., 0.9-1.0 -> 9
    class_labels = (y_continuous * num_classes).floor().long()
    
    # 处理边界情况：值为1.0时应该属于最后一类
    class_labels = torch.clamp(class_labels, 0, num_classes - 1)
    
    return class_labels


