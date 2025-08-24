# 导入部分
import torch
from torch_geometric.data import HeteroData
from torch_geometric.datasets import OGB_MAG
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.loader import NeighborLoader  # 重新添加NeighborLoader
import torch.nn.functional as F
import time
from tqdm import tqdm
import os
import datetime

# ==================== 统一参数配置 ====================
# 训练参数
TOTAL_EPOCHS = 100          # 总训练轮数
BATCH_SIZE = 128            # 批次大小
LEARNING_RATE = 0.01        # 学习率
HIDDEN_CHANNELS = 64        # 隐藏层维度
NUM_NEIGHBORS = [10, 10]    # 邻居采样数量
EVAL_INTERVAL = 10          # 每隔多少个epoch进行一次评估

# 文件路径配置
DATA_ROOT = './data'        # 数据集根目录
SAVED_MODELS_DIR = './saved_models'  # 模型保存目录
LOGS_DIR = './logs'         # 日志保存目录

# 优化器配置
OPTIMIZER_TYPE = 'Adam'     # 优化器类型
# ====================================================

# 设备检测和设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 加载OGB_MAG数据集
print("正在加载OGB_MAG数据集...")
dataset = OGB_MAG(root=DATA_ROOT, preprocess='metapath2vec', transform=T.ToUndirected())
data = dataset[0]

# 获取异构图元数据
node_types, edge_types = data.metadata()
print(f"node_types: {node_types}")
print(f"edge_types: {edge_types}")

print(f"数据集加载完成！")
print(f"数据集名称: {dataset.__class__.__name__}")
print(f"数据集大小: {len(dataset)}")
print(f"data,{data}")

# OGB_MAG数据集信息输出
print("=" * 60)
print("OGB_MAG异构图数据集信息")
print("=" * 60)

# 定义异构图神经网络模型
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# 模型定义
model = GNN(hidden_channels=HIDDEN_CHANNELS, out_channels=dataset.num_classes)
model = to_hetero(model, data.metadata(), aggr='sum')

# 将模型和数据移动到设备
model = model.to(device)
data = data.to(device)

print(f"\n模型信息:")
print(f"模型类型: {type(model)}")
print(f"输出类别数: {dataset.num_classes}")
print(f"模型结构: {model}")

# 创建数据加载器
print("\n创建数据加载器...")

# 训练集加载器
train_loader = NeighborLoader(
    data,
    num_neighbors=NUM_NEIGHBORS,
    batch_size=BATCH_SIZE,
    input_nodes=('paper', data['paper'].train_mask),
    shuffle=True,
)

# 验证集加载器
val_loader = NeighborLoader(
    data,
    num_neighbors=NUM_NEIGHBORS,
    batch_size=BATCH_SIZE,
    input_nodes=('paper', data['paper'].val_mask),
    shuffle=False,
)

# 测试集加载器
test_loader = NeighborLoader(
    data,
    num_neighbors=NUM_NEIGHBORS,
    batch_size=BATCH_SIZE,
    input_nodes=('paper', data['paper'].test_mask),
    shuffle=False,
)

print(f"训练集批次数: {len(train_loader)}")
print(f"验证集批次数: {len(val_loader)}")
print(f"测试集批次数: {len(test_loader)}")
print(f"训练节点数: {data['paper'].train_mask.sum().item()}")
print(f"验证节点数: {data['paper'].val_mask.sum().item()}")
print(f"测试节点数: {data['paper'].test_mask.sum().item()}")
print("数据准备完成！")

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 修改后的训练函数 - 使用NeighborLoader
def train():
    model.train()
    total_examples = total_loss = 0
    
    # 为训练批次添加进度条
    train_bar = tqdm(train_loader, desc='Training', leave=True, 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
    
    for batch in train_bar:
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch['paper'].batch_size
        
        # 前向传播
        out = model(batch.x_dict, batch.edge_index_dict)
        
        # 计算损失（只对目标节点）
        loss = F.cross_entropy(out['paper'][:batch_size], 
                              batch['paper'].y[:batch_size])
        
        loss.backward()
        optimizer.step()
        
        total_examples += batch_size
        total_loss += float(loss) * batch_size
        
        # 更新进度条显示当前批次信息
        current_loss = total_loss / total_examples
        train_bar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Batch_Size': batch_size
        })
    
    return total_loss / total_examples

def test(loader, dataset_name="Evaluating"):
    model.eval()
    total_correct = total_examples = 0
    
    # 为测试批次添加进度条，根据数据集名称显示不同描述
    test_bar = tqdm(loader, desc=f'{dataset_name}', leave=False,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
    
    with torch.no_grad():
        for batch in test_bar:
            batch = batch.to(device)
            batch_size = batch['paper'].batch_size
            
            # 前向传播
            out = model(batch.x_dict, batch.edge_index_dict)
            pred = out['paper'][:batch_size].argmax(dim=-1)
            
            # 计算准确率（只对目标节点）
            correct = (pred == batch['paper'].y[:batch_size]).sum().item()
            total_correct += correct
            total_examples += batch_size
            
            # 更新进度条显示当前批次信息
            current_acc = total_correct / total_examples if total_examples > 0 else 0.0
            test_bar.set_postfix({
                'Acc': f'{current_acc:.4f}',
                'Batch_Size': batch_size
            })
    
    return total_correct / total_examples if total_examples > 0 else 0.0

# 开始训练
print("\n开始训练...")

# 记录训练时间和指标
total_start_time = time.time()
epoch_times = []  # 存储每个epoch的训练时间
epoch_metrics = []  # 存储每个epoch的指标

# 训练循环
epoch_bar = tqdm(range(1, TOTAL_EPOCHS + 1), desc='Overall Progress')

for epoch in epoch_bar:
    # 记录epoch开始时间
    epoch_start_time = time.time()
    
    loss = train()
    
    # 记录epoch结束时间
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    epoch_times.append(epoch_duration)
    
    # 每个epoch都计算指标
    # train_acc = test(train_loader, "Train Eval")
    val_acc = test(val_loader, "Val Eval")
    test_acc = test(test_loader, "Test Eval")
    
    # 存储epoch指标
    epoch_metrics.append({
        'epoch': epoch,
        'loss': loss,
        # 'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'time': epoch_duration
    })
    
    # 更新进度条
    epoch_bar.set_postfix({
        'Epoch': epoch,
        'Loss': f'{loss:.4f}',
        # 'Train': f'{train_acc:.4f}',
        'Val': f'{val_acc:.4f}',
        'Test': f'{test_acc:.4f}',
        'Time': f'{epoch_duration:.1f}s'
    })
    
    if epoch % EVAL_INTERVAL == 0:    
        # 计算平均epoch时间
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        print(f'\nEpoch {epoch:03d}, Loss: {loss:.4f},'
              # f'Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}, '
              f'Epoch Time: {epoch_duration:.2f}s, Avg Time: {avg_epoch_time:.2f}s')

# 训练结束
total_end_time = time.time()
total_training_time = total_end_time - total_start_time

print("\n训练完成！")
print(f"总训练时间: {total_training_time:.2f}秒 ({total_training_time/60:.2f}分钟)")
print(f"平均每个epoch时间: {sum(epoch_times)/len(epoch_times):.2f}秒")
print(f"最快epoch时间: {min(epoch_times):.2f}秒")
print(f"最慢epoch时间: {max(epoch_times):.2f}秒")

# 保存模型
print("\n正在保存模型...")

save_dir = SAVED_MODELS_DIR
os.makedirs(save_dir, exist_ok=True)

# 生成时间戳
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"hetero_gnn_model_{timestamp}.pth"
model_path = os.path.join(save_dir, model_filename)

# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'dataset_info': {
        'num_classes': dataset.num_classes,
        'hidden_channels': HIDDEN_CHANNELS,
        'out_channels': dataset.num_classes
    },
    'training_info': {
        'total_epochs': TOTAL_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'optimizer_type': OPTIMIZER_TYPE,
        'total_training_time': total_training_time,
        'avg_epoch_time': sum(epoch_times)/len(epoch_times),
        'final_loss': loss,
        # 'final_train_acc': train_acc,
        'final_val_acc': val_acc,
        'final_test_acc': test_acc
    },
    'timestamp': timestamp
}, model_path)

print(f"模型已保存到: {model_path}")
print(f"模型文件名: {model_filename}")

# 保存训练日志
print("\n正在保存训练日志...")

logs_dir = LOGS_DIR
os.makedirs(logs_dir, exist_ok=True)

# 生成日志文件名
log_filename = f"training_log_{timestamp}.txt"
log_path = os.path.join(logs_dir, log_filename)

# 写入训练日志
with open(log_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("异构图神经网络训练日志\n")
    f.write("=" * 80 + "\n")
    f.write(f"训练开始时间: {datetime.datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"训练结束时间: {datetime.datetime.fromtimestamp(total_end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"时间戳: {timestamp}\n")
    f.write("\n")
    f.write("设备信息:\n")
    f.write(f"  使用设备: {device}\n")
    if torch.cuda.is_available():
        f.write(f"  GPU名称: {torch.cuda.get_device_name(0)}\n")
        f.write(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    f.write("\n")
    f.write("数据集信息:\n")
    f.write(f"  数据集名称: {dataset.__class__.__name__}\n")
    f.write(f"  数据集大小: {len(dataset)}\n")
    f.write(f"  类别数量: {dataset.num_classes}\n")
    f.write(f"  训练节点数: {data['paper'].train_mask.sum().item()}\n")
    f.write(f"  验证节点数: {data['paper'].val_mask.sum().item()}\n")
    f.write(f"  测试节点数: {data['paper'].test_mask.sum().item()}\n")
    f.write("\n")
    f.write("模型信息:\n")
    f.write(f"  模型类型: {type(model)}\n")
    f.write(f"  隐藏层维度: {HIDDEN_CHANNELS}\n")
    f.write(f"  输出类别数: {dataset.num_classes}\n")
    f.write(f"  训练方式: NeighborLoader批处理训练\n")
    f.write(f"  批次大小: {BATCH_SIZE}\n")
    f.write(f"  邻居采样数: {NUM_NEIGHBORS}\n")
    f.write("\n")
    f.write("\n")
    f.write("训练参数:\n")
    f.write(f"  总epoch数: {TOTAL_EPOCHS}\n")
    f.write(f"  学习率: {LEARNING_RATE}\n")
    f.write(f"  优化器: {OPTIMIZER_TYPE}\n")
    f.write(f"  隐藏层维度: {HIDDEN_CHANNELS}\n")
    f.write(f"  评估间隔: {EVAL_INTERVAL}\n")
    f.write("\n")
    f.write("\n")
    f.write("训练结果:\n")
    f.write(f"  最终损失: {loss:.6f}\n")
    # f.write(f"  最终训练准确率: {train_acc:.6f}\n")
    f.write(f"  最终验证准确率: {val_acc:.6f}\n")
    f.write(f"  最终测试准确率: {test_acc:.6f}\n")
    f.write("\n")
    f.write("\n")
    f.write("时间统计:\n")
    f.write(f"  总训练时间: {total_training_time:.2f}秒 ({total_training_time/60:.2f}分钟)\n")
    f.write(f"  平均每个epoch时间: {sum(epoch_times)/len(epoch_times):.2f}秒\n")
    f.write(f"  最快epoch时间: {min(epoch_times):.2f}秒\n")
    f.write(f"  最慢epoch时间: {max(epoch_times):.2f}秒\n")
    f.write("\n")
    f.write("\n")
    f.write("模型保存信息:\n")
    f.write(f"  模型文件路径: {model_path}\n")
    f.write(f"  模型文件名: {model_filename}\n")
    f.write("\n")
    f.write("\n")
    f.write("每个epoch详细指标记录:\n")
    f.write(f"{'Epoch':>6} {'Loss':>10} {'Val_Acc':>10} {'Test_Acc':>10} {'Time(s)':>8}\n")
    f.write("-" * 60 + "\n")
    for metrics in epoch_metrics:
        f.write(f"{metrics['epoch']:>6} {metrics['loss']:>10.6f}"
                f"{metrics['val_acc']:>10.6f} {metrics['test_acc']:>10.6f} {metrics['time']:>8.2f}\n")
    f.write("\n")
    f.write("\n")
    f.write("每个epoch时间记录:\n")
    for i, epoch_time in enumerate(epoch_times, 1):
        f.write(f"  Epoch {i:3d}: {epoch_time:.2f}秒\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("训练日志记录完成\n")
    f.write("=" * 80 + "\n")

print(f"训练日志已保存到: {log_path}")
print(f"日志文件名: {log_filename}")
print("\n训练和保存完成！")