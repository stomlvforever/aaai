import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import r2_score

def sample_nodes_by_ratio(graph, ratio=0.2, seed=42):
    """
    按照比例从每个子图中采样节点
    Args:
        graph (torch_geometric.data.Data): 包含节点和子图ID的图数据
        ratio (float): 采样比例，默认为 0.2（表示采样 20% 的节点）
        seed (int): 随机种子，用于控制采样的可重复性
    Returns:
        sampled_node_indices (torch.Tensor): 被采样的节点索引
    """
    torch.manual_seed(seed)  # 设置随机种子，确保可重复性

    # 存储采样的节点索引
    sampled_node_indices = []

    # 遍历每个子图的节点
    unique_graph_ids = torch.unique(graph.graph_id)  # 获取所有子图的 ID
    for graph_id in unique_graph_ids:
        # 获取当前子图的所有节点索引
        current_graph_nodes = torch.nonzero(graph.graph_id == graph_id).squeeze()

        # 根据比例计算采样数量
        num_samples = int(len(current_graph_nodes) * ratio)

        # 进行随机采样
        sampled_nodes = torch.randperm(len(current_graph_nodes))[:num_samples]
        sampled_node_indices.append(current_graph_nodes[sampled_nodes])

    # 将所有采样的节点索引拼接在一起
    return torch.cat(sampled_node_indices)


def plot_true_values_distribution_before_sampling(dataset, dataset_name=None, save_dir="distribution_plots"):
    """
    在采样之前绘制真实值的分布直方图
    Args:
        dataset: 数据集对象
        dataset_name: 数据集名称（如 integrated_power_density_prediction_graph）
        save_dir: 保存图片的目录
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
    绘制真实值vs预测值的散点图
    Args:
        pred_values: 预测值数组
        true_values: 真实值数组  
        epoch: 当前epoch数
        split_name: 数据集名称 (train/val/test)
        save_dir: 保存目录，如果为None则自动生成基于PID的目录
        device: 计算设备，如果为None则自动检测
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
    绘制归一化前的数据分布图（支持GPU计算）
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
        median_val = torch.median(data_tensor.flatten().float()).values.item()
        
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