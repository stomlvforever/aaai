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


NET = 0
DEV = 1
PIN = 2


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
        if args.use_stats + self.use_cl == 2:
            assert hidden_dim % 3 == 0, \
                "hidden_dim should be divided by 3 (3 types of encoders)"
            node_embed_dim = hidden_dim // 3

        ## circuit statistics/pe encoder + node&edge type encoders
        elif self.use_stats + self.use_cl == 1:
            assert hidden_dim % 2 == 0, \
                "hidden_dim should be divided by 2 (2 types of encoders)"
            node_embed_dim = hidden_dim // 2

        ## only use node&edge type encoders
        else:
            pass

        ## Contrastive learning encoder
        if self.use_cl:
            self.cl_linear = nn.Linear(args.cl_hid_dim, node_embed_dim)

        ## Circuit Statistics encoder, producing matrix C
        if self.use_stats:
            ## add node_attr transform layer for net/device/pin nodes, by shan
            self.net_attr_layers = nn.Linear(17, node_embed_dim, bias=True)
            self.dev_attr_layers = nn.Linear(17, node_embed_dim, bias=True)
            ## pin attributes are {0, 1, 2} for gate pin, source/drain pin, and base pin
            self.pin_attr_layers = nn.Embedding(17, node_embed_dim)
            self.c_embed_dim = node_embed_dim

        ## Node / Edge type encoders.
        ## Node attributes are {0, 1, 2} for net, device, and pin
        
        # self.node_encoder = nn.Embedding(num_embeddings=4,
        #                                  embedding_dim=node_embed_dim)
        # ## Edge attributes are {0, 1} for 'device-pin' and 'pin-net' edges
        # self.edge_encoder = nn.Embedding(num_embeddings=4,
        #                                  embedding_dim=hidden_dim)
        
        node_type_vocab_size = getattr(args, 'node_type_vocab_size', 4)
        self.node_encoder = nn.Embedding(num_embeddings=node_type_vocab_size,
                                        embedding_dim=node_embed_dim)

        ## Edge attributes - 使用动态大小
        edge_type_vocab_size = getattr(args, 'edge_type_vocab_size', 4)
        self.edge_encoder = nn.Embedding(num_embeddings=edge_type_vocab_size,
                                        embedding_dim=hidden_dim)
        # GNN layers
        self.layers = nn.ModuleList()
        self.model = args.model

        for _ in range(args.num_gnn_layers):
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
            elif args.model == 'gps_attention':
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
        
        self.src_dst_agg = args.src_dst_agg

        ## Add graph pooling layer
        if args.src_dst_agg == 'pooladd':
            self.pooling_fun = pygnn.pool.global_add_pool
            
        elif args.src_dst_agg == 'poolmean':
            self.pooling_fun = pygnn.pool.global_mean_pool
            
        elif args.src_dst_agg == 'globalattn':
            self.attn_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.pooling_fun = AttentionalAggregation(gate_nn=self.attn_nn)
        ## The head configuration
        head_input_dim = hidden_dim * 2 if self.src_dst_agg == 'concat'  and self.task_level == 'edge' else hidden_dim

        if self.task == 'regression':
            dim_out = 1
        elif self.task =='classification':
            dim_out = args.num_classes
        else:
            raise ValueError('Invalid task')
        
        # head MLP layers
        self.head_layers = MLP(
            in_channels=head_input_dim, 
            hidden_channels=hidden_dim, 
            out_channels=dim_out, 
            num_layers=args.num_head_layers, 
            use_bn=use_bn, dropout=dropout, 
            activation=args.act_fn,
        )

        ## Batch normalization
        self.use_bn = args.use_bn
        self.bn_node_x = nn.BatchNorm1d(hidden_dim)
        if self.use_bn and self.use_cl:
            print("[Warning] Using batch normalization with contrastive learning may cause performance degradation.")

        ## activation setting
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
        
        ## Dropout setting
        self.drop_out = args.dropout
        
        if self.task_level == 'node' and args.src_dst_agg == 'globalattn':
            print("[Warning] global attention is not typically used for node-level tasks.")


    def forward(self, batch):
        # print(f"batch.y unique values: {batch.y}")
        # print(f"batch.y[:, 0] unique: {batch.y[:, 0]}")
        # print(f"batch.y[:, 1] unique: {batch.y[:, 1]}")
        
        # # 检查节点类型分布
        # print(f"batch.node_type unique: {batch.node_type}")
        # print(f"node_type distribution: {batch.node_type}")
        
        # # 如果你怀疑是采样问题，可以检查原始图中的标签分布
        # print(f"batch.n_id range: min={batch.n_id.min()}, max={batch.n_id.max()}")
        # # 检查是否所有batch都是这样
        # print(f"Current batch info: batch_size={getattr(batch, 'batch_size', 'unknown')}")
        # print(f"batch.y:{torch.unique(batch.y)}")
        # assert 0
        ## Node type / Edge type encoding
        # print(f"edge_type dtype: {batch.edge_type.dtype}")
        # print(f"batch.node_type.dtype:{batch.node_type.dtype}")
        # assert 0
        x = self.node_encoder(batch.node_type)
        xe = self.edge_encoder(batch.edge_type)
        # print("x", x.size()) #([30672, 72])
        # print("edge_attr",xe.size()) #([61880, 144])
        # assert 0
        ## Contrastive learning encoder
        if self.use_cl:
            xcl = self.cl_linear(batch.x)
            ## concatenate node embeddings and embeddings learned by SGRL
            x = torch.cat((x, xcl), dim=1)

        
        ## If we use circuit statistics encoder
        if self.use_stats:
            net_node_mask = batch.node_type == NET
            dev_node_mask = batch.node_type == DEV
            pin_node_mask = batch.node_type == PIN
            ## circuit statistics embeddings (C in EQ.6)
            node_attr_emb = torch.zeros(
                (batch.num_nodes, self.c_embed_dim), device=batch.x.device
            )
            node_attr_emb[net_node_mask] = \
                self.net_attr_layers(batch.node_attr[net_node_mask])
            node_attr_emb[dev_node_mask] = \
                self.dev_attr_layers(batch.node_attr[dev_node_mask])
            node_attr_emb[pin_node_mask] = \
                self.pin_attr_layers(batch.node_attr[pin_node_mask, 0].long())
            ## concatenate node embeddings and circuit statistics embeddings (C in EQ.6)
            x = torch.cat((x, node_attr_emb), dim=1)
            # print(f"x:{x.shape}")
            # assert 0
            
        # GNN layers
        if self.model == 'gps_attention':
            batch.x = x
            batch.edge_attr = xe
            # print("x", x.size()) #([30672, 72]) 
            # print("edge_attr",batch.edge_attr.size()) #([61880, 144])
            # assert 0
            for layer in self.layers:
                #batch.batch:Data(edge_index=[2, 17734], x=[11919, 144], y=[11919, 2], node_type=[11919], edge_type=[17734], name='ssram', node_attr=[11919, 17], tar_edge_dist=[3], n_id=[11919], e_id=[17734], num_sampled_nodes=[5], num_sampled_edges=[4], input_id=[128], batch_size=128, edge_attr=[17734, 144])
                # print(f"batch:{batch}") 
                # assert 0
                batch = layer(batch)  # GPSLayer 接收整个 batch
            x = batch.x  # 最后提取节点特征
        else:
            # x = x.float()

            for conv in self.layers:
                if self.model == 'gine' or self.model == 'resgatedgcn' :
                    x = conv(x, batch.edge_index, edge_attr=xe)
                elif self.model == 'CustomGatedGCN' or self.model == 'CustomGCNConv' or self.model == 'CustomGINEConv':
                    batch.x = x
                    batch.edge_attr = xe
                    batch = conv(batch)
                    x = batch.x
                else:
                    # print(f"x:{x.shape}") #[28650, 1]
                    # print(f"edge_index:{batch.edge_index.shape}") #([2, 57994])
                    # assert 0
                    x = conv(x, batch.edge_index)

                if self.use_bn:
                    x = self.bn_node_x(x)
                x = self.activation(x)

                if self.drop_out > 0.0:
                    x = F.dropout(x, p=self.drop_out, training=self.training)

            # batch.x = x  # 如果后面还要用 batch


        ## task level : node
        if self.task_level == 'node':
            if self.net_only:
                net_node_mask = batch.node_type == NET
                pred = self.head_layers(x[net_node_mask])
                true_class = batch.y[:, 1][net_node_mask].long()
                true_label = batch.y[net_node_mask]
            else:
                pred = self.head_layers(x)
                true_class = batch.y[:, 1].long()
                true_label = batch.y

            
        elif self.task_level == 'edge':
            # if self.src_dst_agg in ['pooladd', 'poolmean', 'globalattn']:
            #     graph_emb = self.pooling_fun(x, batch.batch)
            if self.src_dst_agg[:4] == 'pool':
                graph_emb = self.pooling_fun(x, batch.batch)
            else:
                batch_size = batch.edge_label.size(0)
                src_emb = x[:batch_size, :]
                dst_emb = x[batch_size:batch_size*2, :]
                if self.src_dst_agg == 'concat':
                    graph_emb = torch.cat((src_emb, dst_emb), dim=1)
                else:
                    graph_emb = src_emb + dst_emb


            pred = self.head_layers(graph_emb)
            true_class = batch.edge_label[:, 1].long()
            true_label = batch.edge_label
        
        else:
            raise ValueError('Invalid task level')
            
        return pred,true_class,true_label
        
        
    

