import argparse
import torch
import numpy as np
from dataset import performat_SramDataset
from downstream_train import downstream_train
import os
import random
import datetime
import sys
#main.py主要操作为：1.定义超参数(parser) 2.加载数据集(performat_SramDataset) 3.训练(downstream_train) 
if __name__ == "__main__":
    # STEP 0: Parse Arguments ======================================================================= #
    parser = argparse.ArgumentParser(description="AAAI")
    # Task setting
    #无论什么输入，对于一个图都有四个基本任务，节点（node）分类与回归，边（edge）分类与回归。
    parser.add_argument("--task_level", type=str, default="node", choices=['node','edge','floorplan'], help="Task level. 'node' or 'edge'.")
    parser.add_argument("--task", type=str, default="regression", help="Task type. 'classification' or 'regression'.")
   
    # Dataset setting
    #这里就是设置数据集，--dataset是数据集名称，通过split(+)分割，然后path.join自动寻找路径。训练时只要把数据集和代码放在一个文件夹下，然后改这里的名字就行
    #--neg_edge_ratio代表负边率，后面会有对应代码即在图中加入多少负边，--net_only代表仅用node节点而抛弃edge特征。看具体任务改，一般保持默认，只不过跑脚本时，注意--neg_edge_ratio的区别
    parser.add_argument("--dataset", type=str, default="integrated_position_prediction_graph",choices=["integrated_power_density_prediction_graph","integrated_position_prediction_graph","integrated_route_with_global_features"], help="Names of datasets.integrated_position_prediction_graph+integrated_power_density_prediction_graph") # the first dataset is the training dataset
    parser.add_argument('--neg_edge_ratio',type=float,default=0.5,help='The ratio of negative edges.') # 0.0 for classification, 0.5 for regression
    parser.add_argument('--net_only',type=int,default=0,help='Only use net nodes for node level task or not.')
    
    # Graph sampling setting
    #这里是加载数据集部分的超参数，加载数据集就是把图导入后，进行一系列操作(如剔除电源节点，加入负边等)，然后--num_hops跳采样，每次跳采样的邻居个数为--num_neighbors，然后把图切分成训练集、验证集、测试集。
    #--small_dataset_sample_rates代表小数据集的采样率，--large_dataset_sample_rates代表大数据集的目标边数，一般默认，看训练效果吧。
    # parser.add_argument("--small_dataset_sample_rates", type=float, default=1.0, help="The sample rate for small dataset.")
    parser.add_argument("--sample_ratio", type=float, default=1, help='子图采样')
    parser.add_argument("--num_hops", type=int, default=2, help="Number of hops in subgraph sampling.")
    parser.add_argument('--num_neighbors',type=int,default=16,help='The number of neighbors in subgraph sampling.')
    
    # Training setting
    #这部分是总体训练的超参数控制，为什么说总体，因为后面还有GNN+的超参数，以及downstream的超参数，分别对应上下游各自的参数，这里是总体的。
    #--seed是随机种子，--num_workers是数据加载的线程数，--gpu是使用的gpu索引，--epochs是训练的轮数，--batch_size是训练的batch大小，--lr是学习率。
    #这里只有batch_size和lr是需要调整的，其他的保持默认即可。batch_size越大，lr也需要相应增大。
    #一般来说，上面说的四个任务，每个任务应该差不多占满一个卡的显存，所以batch_size和lr都需要适当调整。
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--num_workers", type=int, default=8, help="The number of workers in data loaders.")
    parser.add_argument("--gpu", type=int, default=1, help="GPU index. Default: -1, using cpu.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")

    # 删除以下SGRL参数定义：
    # parser.add_argument('--sgrl', type=int, default=0, help='Enable contrastive learning, i.e., SGRL.')
    # parser.add_argument('--e1_lr', type=float, default=1e-6, help='Learning rate for online encoder in SGRL.')
    # parser.add_argument('--e2_lr', type=float, default=2e-7, help='Learning rate for target encoder in SGRL.')
    # parser.add_argument('--momentum', type=float, default=0.99, help='EMA')
    # parser.add_argument('--weight_decay', type=float, default=0., help='weight_decay')
    # parser.add_argument('--cl_epochs', type=int, default=800, help='cl_epochs')
    # parser.add_argument("--cl_model", type=str, default='clustergcn', 
    #                     choices=['clustergcn', 'resgatedgcn', 'gat', 'gcn', 'sage', 'gine'],
    #                     help="The gnn model of SGRL encoders.")
    # parser.add_argument('--cl_num_neighbors', type=int, default=64, 
    #                     help='Number of neighbors for contrastive learning. 128 for large G')
    # parser.add_argument('--cl_dropout', type=float, default=0.3, help='Dropout for SGRL encoders.')
    
    ## global attention setting
    #这里是全局注意力机制的超参数，主要是设置全局注意力机制的类型，以及GNN+的设置。也就是所谓的上游模型，对数据的特征进行加强处理后送人downstream进行分类或回归。
    #--attn_dropout是丢失率，这里要固定0.7，--global_model_type根据之前的实验，选择None是较优的，因为Transformer没有加入pe和se，所以不如不加。
    #--local_gnn_type这里多个模型供选择，后面三个就是自定义的GNN+模型，内部架构由后面的(gnn+ setting)参数控制。
    #--num_heads是多头注意力机制的个数，最优是3，--layer_norm和batch_norm是两种正则化方法，gnn+选择ln较优。
    #但是图变了的话，这些参数应该是要重新调整的。
    parser.add_argument("--attn_dropout", type=float, default=0.7, help='Dropout for attentional networks')
    parser.add_argument("--global_model_type", type=str, default='None', choices=['Transformer' ,'BiasedTransformer', 'None'], help='Global attention mechanism settings')
    parser.add_argument("--local_gnn_type", type=str, default='CustomGatedGCN', choices=['GCN', 'GIN', 'GINE', 'GENConv', 'GAT', 'PNA', 'CustomGatedGCN', 'CustomGCNConv', 'CustomGINEConv'], help='The setting of the global attention GNN layer')
    parser.add_argument("--num_heads", type=int, default=2, help='The number of heads in multi-head attention')
    parser.add_argument("--layer_norm", type=int, default=0, help='Whether to use layer_norm in GNN+, if you use the attention mechanism, layernorm and batchnorm cannot both be equal to 1.')
    parser.add_argument("--batch_norm", type=int, default=1, help='Whether to use batch_norm in GNN+, if you use the attention mechanism, layernorm and batchnorm cannot both be equal to 1.')
    
    ## gnn+ setting
    #这里是GNN+的超参数，如果上面--global_model_type选择了后面三个自定义的模型，这里就是选择在模型里是否使用残差连接，归一化，正则化，以及前馈网络，要调整的话得做消融实验，但一般来说，保持默认即可。
    parser.add_argument('--residual', type=int, default=1, help='Whether to use residuals in GNN+')
    parser.add_argument('--g_bn', type=int, default=1, help='Whether to use bn in GNN+')
    parser.add_argument('--g_drop', type=float, default=0.3, help='Whether to use bn in GNN+')
    parser.add_argument('--g_ffn', type=int, default=1, help='Whether to use ffn in GNN+')
    
    ## Downstream GNN setting
    #数据过了上游GNN+以及Transformer后，更新的数据特征送到下游的GNN模型训练。
    #--model就是选择模型类别，只有选择gps_attention，前面讲的上游模型才起作用，选其他的，数据直接送人Downstream GNN。
    #--num_gnn_layers就是模型层数，默认即可,越大跑得越慢。
    #--num_head_layers就是多头注意力机制的层数，默认2，一般保持默认即可。
    #--hid_dim就是隐藏层维度，默认144，一般保持默认即可。
    #--dropout是丢失率，默认0.3，一般保持默认即可。
    #上面默认的参数都是baseline或其他实验跑出来的较优值，实际上有时间应该调整的。
    #--use_bn是是否使用bn，建议0，但是用不用似乎影响不大，--act_fn选择'relu'。--use_stats保持默认。
    parser.add_argument("--model", type=str, default='gps_attention', choices=['clustergcn', 'resgatedgcn', 'gat', 'gcn', 'sage', 'gine', 'gps_attention'], help="The gnn model. Could be 'clustergcn', 'resgatedgcn', 'gat', 'gcn', 'sage', 'gine', 'gps_attention'.")
    parser.add_argument("--num_gnn_layers", type=int, default=4, help="Number of GNN layers.")
    parser.add_argument("--num_head_layers", type=int, default=2, help="Number of head layers.")
    parser.add_argument("--hid_dim", type=int, default=16, help="Hidden layer dim.")
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout for neural networks.')
    parser.add_argument('--use_bn', type=int, default=0, help='0 or 1. Batch norm for neural networks.')
    parser.add_argument('--act_fn', default='prelu', choices=['relu', 'elu', 'tanh', 'leakyrelu', 'prelu'], help='Activation function')
    parser.add_argument('--use_stats', type=int, default=0, help='0 or 1. Circuit statistics features. Use in node task.')

    # Regression setting
    #这是对于回归任务的单独设置了，因为分类回归的指标和损失函数不同，所以在这里选。基本不用改，或者根据在跑的实验改。
    parser.add_argument('--src_dst_agg', type=str, default='concat',choices=['concat', 'add', 'pooladd', 'poolmean', 'globalattn'],help="The way to aggregate nodes.")
    parser.add_argument("--regress_loss", type=str, default='mse', choices=['mse', 'gai', 'bmc', 'bni', 'lds'], help="The loss function for edge regression. Could be 'mse', 'bmc', or 'gai'.")
    
    # Classification setting
    parser.add_argument('--class_loss',type=str,default='cross_entropy',choices=['bsmCE','focal','cross_entropy'],help='The loss function for classification.')
    parser.add_argument('--num_classes',type=int,default=5,help='The number of classes for node classification.')
    parser.add_argument('--class_boundaries',type=list,default=[0.2,0.4,0.6,0.8],help='The boundaries for classification.')

    # Balanced MSE setting for GAI implementation
    parser.add_argument("--noise_sigma", type=float, default=0.001, help="The simga_noise of Balanced MSE (EQ 3.6).")
    
    # LDS setting
    parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
    parser.add_argument('--lds_ks', type=int, default=9, help='LDS kernel size: should be odd number. 5 0r 9.')
    parser.add_argument('--lds_sigma', type=float, default=0.02, help='LDS gaussian/laplace kernel sigma. 1 or 2.')
    
    parser.add_argument('--log_dir', type=str, default='logs', help='The directory to save the log file.')

    args = parser.parse_args()

    # Syncronize all random seeds
    #随机种子就是保证实验可重复性的，了解就好
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #set log file
    
    # create log file
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.task == 'classification':
        log_filename = os.path.join(args.log_dir, f"{timestamp}_{args.task_level}_{args.task}_{args.dataset}_loss{args.class_loss}_batch{args.batch_size}_{args.src_dst_agg}.txt")
    else: # regression task
        log_filename = os.path.join(args.log_dir, f"{timestamp}_{args.task_level}_{args.task}_{args.dataset}_loss{args.regress_loss}_batch{args.batch_size}_{args.src_dst_agg}.txt")
    log_file = open(log_filename, 'w')
    
    # Redirect standard output to both file and console
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)


    # Check cuda
    #训练从这里开始，首先根据--gpu的值选择对应的卡，然后检查是否可用
    if args.gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
        print('Using GPU: {}'.format(args.gpu))
    else:
        device = torch.device("cpu")

    print(f"============= PID = {os.getpid()} ============= ")
    print(args)


    # STEP 1: Load Dataset =================================================================== #
    #这里就是加载数据集，首先根据--dataset的值，找到对应的路径，然后加载数据集。
    if args.dataset == 'integrated_position_prediction_graph':
        # 只加载位置预测图
        _, regression_dataset = performat_SramDataset(
            name='integrated_position_prediction_graph',
            dataset_dir='/home/zoujj/project/aaai/datasets/',
            args = args,
            neg_edge_ratio=args.neg_edge_ratio,
            to_undirected=True,
            task_level=args.task_level
        )
    elif args.dataset == 'integrated_power_density_prediction_graph':
        # 只加载功率密度预测图
        _, regression_dataset = performat_SramDataset(
            name='integrated_power_density_prediction_graph',
            dataset_dir='/home/zoujj/project/aaai/datasets/',
            args = args,
            neg_edge_ratio=args.neg_edge_ratio,
            to_undirected=True,
            task_level=args.task_level
        )
    elif args.dataset == 'integrated_route_with_global_features':
        # 只加载功率密度预测图
        _, regression_dataset = performat_SramDataset(
            name='integrated_route_with_global_features',
            dataset_dir='/home/zoujj/project/aaai/datasets/',
            args = args,
            neg_edge_ratio=args.neg_edge_ratio,
            to_undirected=True,
            task_level=args.task_level
        )       
    elif args.dataset == 'integrated_floorplan_area_prediction_graph':
            # 只加载功率密度预测图
            _, regression_dataset = performat_SramDataset(
                name='integrated_floorplan_area_prediction_graph',
                dataset_dir='/home/zoujj/project/aaai/datasets/',
                args = args,
                neg_edge_ratio=args.neg_edge_ratio,
                to_undirected=True,
                task_level=args.task_level
            )           
    # STEP 4: Training Epochs ================================================================ #
    #正式开始训练，args就是参数设置，dataset输入数据，device就是运行设备。

    downstream_train(args, regression_dataset, device)        

    # sys.stdout = original_stdout
    # log_file.close()
    # print(f"Finished running and save results to {log_filename}")