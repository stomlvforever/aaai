import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, ResGatedGraphConv, 
    GINEConv, ClusterGCNConv
)
from torch_geometric.nn.models.mlp import MLP
from torch_geometric.nn.aggr import AttentionalAggregation
from gps_layer import GPSLayer


# NET = 0
# DEV = 1
# PIN = 2


class GraphHead(nn.Module):
    """ GNN head for graph-level prediction.

    Implementation adapted from the transductive GraphGPS.

    Args:
        hidden_dim (int): Hidden features' dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        num_layers (int): Number of layers of GNN model
        layers_post_mp (int): number of layers of head MLP
        use_bn (bool): whether to use batch normalization
        drop_out (float): dropout rate
        activation (str): activation function
        src_dst_agg (str): the way to aggregate src and dst nodes, which can be 'concat' or 'add' or 'pool'
    """
    def __init__(self, args):
        super().__init__()
        self.use_cl = False
        self.use_stats = args.use_stats
        hidden_dim = args.hid_dim
        node_embed_dim = hidden_dim
        self.task = args.task
        self.task_level = args.task_level
        self.net_only = args.net_only
        self.num_classes = args.num_classes
        self.class_boundaries = args.class_boundaries
        # 添加这一行
        self.dataset_name = args.dataset
        local_gnn_type = args.local_gnn_type
        global_model_type = args.global_model_type
        act_fn = args.act_fn
        attn_dropout = args.attn_dropout
        use_bn = args.use_bn
        num_heads = args.num_heads
        dropout = args.dropout
        residual = args.residual
        g_bn = args.g_bn
        g_drop = args.g_drop
        g_ffn = args.g_ffn
        layer_norm = args.layer_norm
        batch_norm = args.batch_norm
        task_level = args.task_level
        
        ## circuit statistics encoder + PE encoder + node&edge type encoders
        #确保隐藏维度层数被三整除并将它平均分配给三种编码器
        # if args.use_stats + self.use_cl == 2:
        #     assert hidden_dim % 3 == 0, \
        #         "hidden_dim should be divided by 3 (3 types of encoders)"
        #     node_embed_dim = hidden_dim // 3

        # ## circuit statistics/pe encoder + node&edge type encoders
        # #同理
        # elif self.use_stats + self.use_cl == 1:
        #     assert hidden_dim % 2 == 0, \
        #         "hidden_dim should be divided by 2 (2 types of encoders)"
        #     node_embed_dim = hidden_dim // 2

        # ## only use node&edge type encoders
        # else:
        #     pass

        # ## Contrastive learning encoder
        # #图对比学习用到的编码器
        # if self.use_cl:
        #     self.cl_linear = nn.Linear(args.cl_hid_dim, node_embed_dim)

        ## Circuit Statistics encoder, producing matrix C
        # if self.use_stats:#为不同类型节点定义不同的编码器，输入维度17，输出维度node_embed_dim
        #     ## add node_attr transform layer for net/device/pin nodes, by shan
        #     self.net_attr_layers = nn.Linear(17, node_embed_dim, bias=True)
        #     self.dev_attr_layers = nn.Linear(17, node_embed_dim, bias=True)
        #     ## pin attributes are {0, 1, 2} for gate pin, source/drain pin, and base pin
        #     self.pin_attr_layers = nn.Embedding(17, node_embed_dim)
        #     self.c_embed_dim = node_embed_dim
        
        #定义节点和边的编码器，将节点类型和边类型编码为向量
        ## Node / Edge type encoders.
        ## Node attributes are {0, 1, 2} for net, device, and pin
        # 在 __init__ 方法中，node_encoder 定义之后添加
        if self.task_level == 'node':
            node_type_vocab_size = getattr(args, 'node_type_vocab_size', 142)
            self.node_encoder = nn.Embedding(num_embeddings=node_type_vocab_size, embedding_dim=node_embed_dim)
            
            # 添加第四列和第六列的 embedding 层
            col4_vocab_size = getattr(args, 'col4_vocab_size', 8)  # 根据实际数据调整
            col6_vocab_size = getattr(args, 'col6_vocab_size', 2)  # 根据实际数据调整
            self.col4_encoder = nn.Embedding(num_embeddings=col4_vocab_size, embedding_dim=node_embed_dim)
            self.col6_encoder = nn.Embedding(num_embeddings=col6_vocab_size, embedding_dim=node_embed_dim)
            self.edge_encoder = nn.Embedding(num_embeddings=4, embedding_dim=node_embed_dim)       
            concatenated_dim = 3 * node_embed_dim + 11
            self.feature_projection = nn.Linear(concatenated_dim, node_embed_dim)
            
        # GNN layers
        self.layers = nn.ModuleList()
        self.model = args.model
        #选择MPNN模型，详见aspdac
        for _ in range(args.num_gnn_layers): #选择一般训练模型
            ## the following are examples of using different GNN layers
            if args.model == 'clustergcn':
                self.layers.append(ClusterGCNConv(hidden_dim, hidden_dim))
            elif args.model == 'gcn':
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            elif args.model == 'sage':
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            elif args.model == 'gat':
                self.layers.append(GATConv(hidden_dim, hidden_dim, heads=1))
            elif args.model == 'resgatedgcn':
                self.layers.append(ResGatedGraphConv(hidden_dim, hidden_dim, edge_dim=hidden_dim))
            elif args.model == 'gine':
                mlp = MLP(
                    in_channels=hidden_dim, 
                    hidden_channels=hidden_dim, 
                    out_channels=hidden_dim, 
                    num_layers=2, 
                    norm=None,
                )
                self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_dim))
            # elif args.model == 'CustomGatedGCN':
            #     self.layers.append(GatedGCNLayer(in_dim=hidden_dim, 
            #                                     out_dim=hidden_dim,
            #                                 dropout=g_drop,
            #                                 residual=residual,
            #                                 ffn=g_ffn,
            #                                 batch_norm=g_bn,
            #                                 ))
            # elif args.model == 'CustomGCNConv':
            #     self.layers.append(GCNConvLayer(dim_in=hidden_dim, 
            #                                     dim_out=hidden_dim,
            #                                 dropout=g_drop,
            #                                 residual=residual,
            #                                 ffn=g_ffn,
            #                                 batch_norm=g_bn,
            #                                 ))
            # elif args.model == 'CustomGINEConv':
            #     self.layers.append(GINEConvLayer(dim_in=hidden_dim, 
            #                                     dim_out=hidden_dim,
            #                                 dropout=g_drop,
            #                                 residual=residual,
            #                                 ffn=g_ffn,
            #                                 batch_norm=g_bn))
            elif args.model == 'gps_attention': #启用上游图注意力机制
                self.layers.append(GPSLayer(hid_dim=hidden_dim, 
                                            local_gnn_type=local_gnn_type,
                                            global_model_type=global_model_type, 
                                            act=act_fn, 
                                            attn_dropout=attn_dropout, 
                                            batch_norm=batch_norm,
                                            layer_norm=layer_norm,
                                            num_heads=num_heads,
                                            residual=residual,
                                            g_bn=g_bn,
                                            g_drop=g_drop,
                                            g_ffn=g_ffn,
                                            task_level=task_level
                                            ))
            else:
                raise ValueError(f'Unsupported GNN model: {args.model}')
        
        self.src_dst_agg = args.src_dst_agg #指定聚合方式

        ## Add graph pooling layer
        if args.src_dst_agg == 'pooladd': 
            self.pooling_fun = pygnn.pool.global_add_pool #定义聚合为加和
            
        elif args.src_dst_agg == 'poolmean':
            self.pooling_fun = pygnn.pool.global_mean_pool #定义聚合为平均值
            
        elif args.src_dst_agg == 'globalattn': #定义全局注意力机制
            self.attn_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.pooling_fun = AttentionalAggregation(gate_nn=self.attn_nn)
        ## The head configuration
        head_input_dim = hidden_dim * 2 if self.src_dst_agg == 'concat'  and self.task_level == 'edge' else hidden_dim #如果嵌入边特征，则输入维度为hidden_dim*2，否则为hidden_dim
        
        #定义模型输出维度
        if self.task == 'regression':
            dim_out = 1
        elif self.task =='classification':
            dim_out = args.num_classes
        else:
            raise ValueError('Invalid task')
        
        # head MLP layers 定义MLP，接收GNN+的输出，输出维度为1或num_classes，详见aspdac
        self.head_layers = MLP(
            in_channels=head_input_dim, 
            hidden_channels=hidden_dim, 
            out_channels=dim_out, 
            num_layers=args.num_head_layers, 
            use_bn=use_bn, dropout=dropout, 
            activation=args.act_fn,
        )
        
        ## Batch normalization #层归一化
        self.use_bn = args.use_bn
        self.bn_node_x = nn.BatchNorm1d(hidden_dim)
        if self.use_bn and self.use_cl:
            print("[Warning] Using batch normalization with contrastive learning may cause performance degradation.")

        ## activation setting #定义激活函数
        if args.act_fn == 'relu':
            self.activation = nn.ReLU()
        elif args.act_fn == 'elu':
            self.activation = nn.ELU()
        elif args.act_fn == 'tanh':
            self.activation = nn.Tanh()
        elif args.act_fn == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif args.act_fn == 'prelu':
            self.activation = nn.PReLU()
        else:
            raise ValueError('Invalid activation')
        
        ## Dropout setting 正则化
        self.drop_out = args.dropout
        
        if self.task_level == 'node' and args.src_dst_agg == 'globalattn':
            print("[Warning] global attention is not typically used for node-level tasks.")

        #前向传播
    def forward(self, batch):
        if self.task_level == 'node':
            if self.dataset_name == 'integrated_position_prediction_graph':
                # 提取需要embedding的离散特征列（注意：Python索引从0开始）
                node_type_ids = batch.x[:, 2].long()  # 第3维：节点类型
                col4_ids = batch.x[:, 3].long()       # 第4维：需要embedding的特征
                col6_ids = batch.x[:, 5].long()       # 第6维：需要embedding的特征
                edge_type_ids = batch.edge_attr[batch.e_id, 1].long()
                
                # 对各个离散特征进行重映射
                unique_node_types, node_inverse = torch.unique(node_type_ids, return_inverse=True)
                unique_col4_types, col4_inverse = torch.unique(col4_ids, return_inverse=True)
                unique_col6_types, col6_inverse = torch.unique(col6_ids, return_inverse=True)
                unique_edge_types, edge_inverse = torch.unique(edge_type_ids, return_inverse=True)
                
                # # 🔍 打印重映射信息
                # print(f"🔍 重映射信息:")
                # print(f"  unique_node_types数量: {len(unique_node_types)}, 最大重映射ID: {node_inverse.max()}")
                # print(f"  unique_col4_types数量: {len(unique_col4_types)}, 最大重映射ID: {col4_inverse.max()}")
                # print(f"  unique_col6_types数量: {len(unique_col6_types)}, 最大重映射ID: {col6_inverse.max()}")
                # print(f"  unique_edge_types数量: {len(unique_edge_types)}, 最大重映射ID: {edge_inverse.max()}")
                
                # 创建重映射的ID
                remapped_node_type_ids = node_inverse.to(batch.x.device)
                remapped_col4_ids = col4_inverse.to(batch.x.device)
                remapped_col6_ids = col6_inverse.to(batch.x.device)
                remapped_edge_type_ids = edge_inverse.to(batch.x.device)
                
                # 通过embedding层编码离散特征
                x_node = self.node_encoder(remapped_node_type_ids)  # embedding维度
                x_col4 = self.col4_encoder(remapped_col4_ids)       # embedding维度
                x_col6 = self.col6_encoder(remapped_col6_ids)       # embedding维度
                xe = self.edge_encoder(remapped_edge_type_ids)

                # 拼接所有特征：连续特征 + 三个embedding特征
                continuous_features = torch.cat([
                    batch.x[:, :2],      # 第0、1维
                    batch.x[:, 4:5],     # 第4维（索引4，实际第5维）
                    batch.x[:, 6:]       # 第6维之后的所有维度
                ], dim=1)
                
                x = torch.cat([continuous_features, x_node, x_col4, x_col6], dim=1)
                # print(f"  batch.x形状: {x.shape}")
                x = self.feature_projection(x)
        
            # elif self.args.dataset == 'integrated_power_density_prediction_graph':
        # GNN layers
            # print(f"  batch.x形状: {x.shape}")
            # print(f"  batch.edge_attr形状: {xe.shape}")
        if self.model == 'gps_attention':
            batch.x = x
            batch.edge_attr = xe
            # print(f"🔍 进入GPS层前:")
            # print(f"  batch.x形状: {batch.x.shape}")
            # print(f"  batch.edge_attr形状: {batch.edge_attr.shape}")
            # print(f"  batch.edge_index形状: {batch.edge_index.shape}")
            # print(f"  边数量: {batch.edge_index.shape[1]}, 边属性数量: {batch.edge_attr.shape[0]}")
    
            for i, layer in enumerate(self.layers):
                # print(f"🔍 GPS层 {i+1} 处理前 batch.x形状: {batch.x.shape}")
                batch = layer(batch)
                # print(f"🔍 GPS层 {i+1} 处理后 batch.x形状: {batch.x.shape}")
            x = batch.x
            # print(f"🔍 所有GPS层处理完成后最终x形状: {x.shape}")
            # assert 0
        else:
            # x = x.float()
            #不用注意力使用普通的GNN
            for conv in self.layers:
                if self.model == 'gine' or self.model == 'resgatedgcn' :
                    x = conv(x, batch.edge_index, edge_attr=xe)
                elif self.model == 'CustomGatedGCN' or self.model == 'CustomGCNConv' or self.model == 'CustomGINEConv':
                    batch.x = x
                    batch.edge_attr = xe
                    batch = conv(batch)
                    x = batch.x
                else:
                    x = conv(x, batch.edge_index)
                #归一化，激活与正则化
                if self.use_bn:
                    x = self.bn_node_x(x)
                x = self.activation(x)

                if self.drop_out > 0.0:
                    x = F.dropout(x, p=self.drop_out, training=self.training)

        #下游模型，利用MLP分类或预测
        ## task level : node
        if self.task_level == 'node':
            # if self.net_only:
            #     net_node_mask = batch.node_type == NET
            #     pred = self.head_layers(x[net_node_mask])
            #     true_class = batch.y[:, 1][net_node_mask].long()
            #     true_label = batch.y[net_node_mask]
            # else:
            print(f"x.size:{x.size()}")
            print(f"x:{x}")
            
            pred = self.head_layers(x)
            # 删除以下4行调试打印语句
            print(f"🔍 下游模型输出 pred 形状: {pred.shape}")
            print(f"🔍 下游模型输出 pred 内容: {pred}")
            # print(f"batch.y:{batch.y}")
            # print(f"batch.y.size:{batch.y.size()}")
            assert 0
            true_class = batch.y.long()
            true_label = batch.y

        elif self.task_level == 'edge': 
            pred = self.head_layers(x)
            true_class = batch.y.long()
            true_label = batch.y
        # elif self.task_level == 'edge':
        #     if self.src_dst_agg[:4] == 'pool': #训练前先池化，一般就是concat而不是pool
        #         graph_emb = self.pooling_fun(x, batch.batch)
        #     else:
        #         batch_size = batch.edge_label.size(0)
        #         src_emb = x[:batch_size, :]
        #         dst_emb = x[batch_size:batch_size*2, :]
        #         if self.src_dst_agg == 'concat':
        #             graph_emb = torch.cat((src_emb, dst_emb), dim=1)
        #         else:
        #             graph_emb = src_emb + dst_emb

        #     #获得预测输出，并返回真实值供后续计算损失
        #     pred = self.head_layers(graph_emb)
        #     true_class = batch.edge_label[:, 1].long()
        #     true_label = batch.edge_label
        
        else:
            raise ValueError('Invalid task level')
        
        # print(f"pred:{pred},true_class:{true_class},true_label:{true_label}")
        # print(f"pred:{pred},true_label:{true_label}")
        # pred_squeezed = pred.squeeze()

        # # 计算绝对误差
        # absolute_error = torch.abs(pred_squeezed - true_label)
        # print(f"绝对误差形状: {absolute_error.shape}")  # [N]
        # print(f"平均绝对误差 (MAE): {torch.mean(absolute_error).item():.4f}")       
        # print(f"pred_squeezed 范围: {pred_squeezed.min():.4f} ~ {pred_squeezed.max():.4f}")
        # print(f"true_label 范围: {true_label.min():.4f} ~ {true_label.max():.4f}") 
        # assert 0
        return pred,true_class,true_label
    

