import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score,
    mean_absolute_error, mean_squared_error,
    root_mean_squared_error, r2_score,
)
from gate_position_evaluator import evaluate_predictions
from gate_power_density_evaluator import evaluate_predictions_res
from via_count_evaluator import evaluate_predictions_via
import numpy as np
import time
from tqdm import tqdm
from model import GraphHead
from sampling import dataset_sampling
# from balanced_mse import GAILoss, BMCLoss, BNILoss, train_gmm, WeightedMSE, get_lds_weights, BalancedSoftmax, FocalLoss, compute_class_weights
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
# from torch.utils.data.sampler import SubsetRandomSampler
# from sram_dataset import LinkPredictionDataset
# from sram_dataset import collate_fn, adaption_for_sgrl
# from torch_geometric.data import Batch

# from torch_geometric.loader import NeighborLoader, GraphSAINTRandomWalkSampler, GraphSAINTEdgeSampler, ShaDowKHopSampler

NET = 0
DEV = 1
PIN = 2
#logfer类记录训练过程的结果

class Logger(object):
    """ 
    Logger for printing message during training and evaluation. 
    Adapted from GraphGPS 
    """
    
    def __init__(self, task='classification', max_label=None, dataset_name=None):
        super().__init__()
        self.test_scores = False
        self._iter = 0
        self._true = []
        self._pred = []
        self._loss = 0.0
        self._size_current = 0
        self.task = task
        self.max_label = max_label
        self.dataset_name = dataset_name  # Store args object
        
    def update_stats(self, true, pred, batch_size, loss):
        self._true.append(true)        
        self._pred.append(pred)        
        self._size_current += batch_size
        self._loss += loss * batch_size  
        self._iter += 1                  

    def write_epoch(self, split=""):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        true = torch.tensor(true)
        pred_score = torch.tensor(pred_score)
        
        # 计算平均loss
        avg_loss = self._loss / self._size_current
        
        metrics = {}
        # Use args.dataset to decide which evaluation function to use
        if self.dataset_name == 'integrated_position_prediction_graph':
            # Use classification evaluation
            metrics = evaluate_predictions(pred_score, true)
        elif self.dataset_name == 'integrated_power_density_prediction_graph':
            # Use regression evaluation
            metrics = evaluate_predictions_res(pred_score, true)
        elif self.dataset_name == 'integrated_route_with_global_features':
            metrics = evaluate_predictions_via(pred_score, true)
        
        # 将loss添加到metrics中
        metrics['loss'] = avg_loss
            
        # 注释掉单个batch损失的输出
        output = f"{split.capitalize()} Metrics: "
        output += ", ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
        print(output)  # 现在包含loss指标
        return metrics

#损失函数
def compute_loss(args, pred, true, criterion):
    """Compute loss and prediction score. 
    Args:
        args (argparse.Namespace): The arguments
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Ground truth label
        criterion (torch.nn.Module): The loss function
    Returns: Loss, normalized prediction score
    """
    assert criterion, "Loss function is not provided!"
    assert pred.size(0) == true.size(0), \
        "Prediction and true label size mismatch!"

    if args.task == 'classification':
        loss = F.cross_entropy(pred, true)
        predict_class = torch.argmax(pred, dim=1)
        return loss, predict_class, true

    elif args.task == 'regression':
        # 确保pred是[N, 1]形状
        if pred.ndim == 1:
            pred = pred.view(-1, 1)
        elif pred.size(1) != 1:
            pred = pred.view(-1, 1)
        
        # 修复维度检查问题
        if true.ndim == 1:
            # 如果true是1D张量，将其转换为2D [N, 1]
            true = true.view(-1, 1)
            # 创建占位符的第二列（全零）
            placeholder = torch.zeros_like(true)
            true_with_placeholder = torch.cat([true, placeholder], dim=1)
        elif true.size(1) == 1:
            # 如果true是[N, 1]，添加第二列占位符
            placeholder = torch.zeros_like(true)
            true_with_placeholder = torch.cat([true, placeholder], dim=1)
        else:
            # 如果true已有两列，直接使用
            assert true.size(1) == 2, \
                "true label should have two columns [continuous label, discrete label or label weights]!"
            true_with_placeholder = true
       
        ## for LDS loss, the second column of `true` is the weights
        # if args.regress_loss == 'lds':
        #     loss = criterion(
        #         pred, 
        #         true_with_placeholder[:, 0].squeeze(), 
        #         true_with_placeholder[:, 1].squeeze()  # the weight for each label
        #     )
            return loss, pred, true_with_placeholder[:, 0].view(pred.size())

        ## for other loss func
        true = true_with_placeholder[:, 0]  # 只使用第一列（真实值）
        true = true.view(-1, 1)  # 确保true是[N, 1]形状
        # print(f"pred:{pred},true:{true},criterion(pred, true):{criterion(pred, true)}")
        # assert 0
        return criterion(pred, true), pred, true
    
    else:
        raise ValueError(f"Task type {args.task} not supported!")

@torch.no_grad()
def eval_epoch(args, loader, model, device, 
               split='val', criterion=None):
    """ 
    evaluate the model on the validation or test set
    Args:
        args (argparse.Namespace): The arguments
        loader (torch.utils.data.DataLoader): The data loader
        model (torch.nn.Module): The model
        device (torch.device): The device to run the model on
        split (str): The split name, 'val' or 'test'
        criterion (torch.nn.Module): The loss function
    """
    model.eval()#开启评估模型，此时不训练仅预测评估，用于测试集
    time_start = time.time()
    logger = Logger(task=args.task,dataset_name=args.dataset) #实例化日志记录器

    for i, batch in enumerate(tqdm(loader, desc="eval_"+split, leave=False)):#遍历loader
        pred, class_true, label_true = model(batch.to(device))#模型预测并获得预测值和真实标签

        if args.task == 'regression':
            loss, pred_score, true = compute_loss(args, pred, label_true, criterion=criterion)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
            # print(f"pred:{_pred},true:{_true},loss:{loss}")
            # assert 0
            logger.update_stats(true=_true,
                                pred=_pred,
                                batch_size=_true.size(0),
                                loss=loss.detach().cpu().item(),
                                )
        elif args.task == 'classification':
            loss, predict_class, true = compute_loss(args, pred, class_true, criterion=criterion)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = predict_class.detach().to('cpu', non_blocking=True)
            logger.update_stats(true=_true,
                                pred=_pred,
                                batch_size=_true.size(0),
                                loss=loss.detach().cpu().item(),
                                )
    return logger.write_epoch(split)
#回归任务训练模块
def regress_train(args, regressor, optimizer, criterion,
          train_loader, val_loader, test_loaders, max_label,
          device, scheduler=None):
    """
    Train the head model for regression task
    Args:
        args (argparse.Namespace): The arguments
        regressor (torch.nn.Module): The regressor
        optimizer (torch.optim.Optimizer): The optimizer
        criterion (torch.nn.Module): The loss function
        train_loader (torch.utils.data.DataLoader): The training data loader
        val_loader (torch.utils.data.DataLoader): The validation data loader  
        test_laders (list): A list of test data loaders
        device (torch.device): The device to train the model on
    """
    optimizer.zero_grad() #清空梯度
    
    best_results = { #初始化最佳结果
        'best_val_mse': 1e9, 'best_val_loss': 1e9, 
        'best_epoch': 0, 'test_results': []
    }
    for epoch in range(args.epochs):#遍历epoch
        logger = Logger(task=args.task, max_label=max_label)#开启日志记录器
        regressor.train()#开启训练模式
        # print(f"你好")
        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):#遍历loader
            optimizer.zero_grad()

            ## Get the prediction from the model
            #训练模型
            # print(f"你好吗")
            y_pred,true_class, y = regressor(batch.to(device))
            y_pred = y_pred.float()
            y = y.float()
            loss, pred, true = compute_loss(args, y_pred, y, criterion=criterion)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = y_pred.detach().to('cpu', non_blocking=True)
            #后向传播并更新梯度
            loss.backward()
            optimizer.step()
            
            ## Update the logger and print message to the screen
            #记录训练过程
            logger.update_stats(
                true=_true, pred=_pred, 
                batch_size=_true.squeeze().size(0), 
                loss=loss.detach().cpu().item()
            )
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")            
        train_res = eval_epoch(
            args, train_loader, 
            regressor, device, split='train', criterion=criterion
        )    
        

        # train_metrics = logger.write_epoch(split='train')  # 打印训练集结果
        # print(f"Training Metrics: {train_res}")
        
        ## ========== validation ========== ##
        val_res = eval_epoch( #获得验证集结果
            args, val_loader, 
            regressor, device, split='val', criterion=criterion
        )
        
        # print(f"Validation Metrics: {val_res}")
        
        if scheduler is not None:
            scheduler.step()
            
        ## update the best results so far
        
        #只有获得更优验证集结果时才在测试集上评估并更新最佳结果
        # if  best_results['best_val_mae'] > val_res['mae']:
        #     best_results['best_val_mae'] = val_res['mae']
        #     best_results['best_val_loss'] = val_res['loss']
        #     best_results['best_epoch'] = epoch
        
        test_results = []
           
            ## ========== testing on other datasets ========== ##
            #在测试集上测试
        for test_name in test_loaders.keys():
            res = eval_epoch(
                args, test_loaders[test_name], 
                regressor, device, split='test', 
                criterion=criterion
            )
            # test_results.append(res)
        os.makedirs("downstream_model", exist_ok=True)
        torch.save(regressor.state_dict(), f"downstream_model/model_{epoch}-{args.regress_loss}.pth")

        # if best_results['best_epoch'] == epoch:
        #     best_results['test_results'] = test_results

        # print( "=====================================")
        # print(f" Best epoch: {best_results['best_epoch']}, mse: {best_results['best_val_mse']}, loss: {best_results['best_val_loss']}")
        # print(f" Test results: {[res for res in best_results['test_results']]}")
        # print( "=====================================")

#分类任务训练模块
def class_train(args, classifier,optimizer_classifier, 
          train_loader, val_loader, test_loaders, max_label,
          device, scheduler=None):
    """
    Train model for capacitance classification task
    Args:
        args (argparse.Namespace): The arguments
        classifier (torch.nn.Module): The classifier
        optimizer_classifier (torch.optim.Optimizer): The optimizer for the classifier
        train_loader (torch.utils.data.DataLoader): The training data loader
        val_loader (torch.utils.data.DataLoader): The validation data loader  
        test_laders (list): A list of test data loaders
        device (torch.device): The device to train the model on
    """
    # Reset optimizers
    optimizer_classifier.zero_grad()

    # create the directory to save the model
    classifier_save_dir = os.path.join("models_node_cap_classifier")
    
    os.makedirs(classifier_save_dir, exist_ok=True)
    
    # initialize the best model metrics

    best_f1 = 0.0
    #大致流程与回归任务类似，不同之处在于计算损失函数
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # add the logger for classification task
        logger = Logger(task='classification', max_label=max_label)
        classifier.train()
        
        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
            # Move batch to device
            batch = batch.to(device)
            optimizer_classifier.zero_grad()
            ## Get the prediction from the model
            class_logits,true_class, true_label = classifier(batch)
            class_probs = F.softmax(class_logits, dim=1)

            predict_class = torch.argmax(class_probs, dim=1)
            
            # 🔍 调试打印 - 看看这里的数据状态
            
            ## set the loss function for classification task
            # if args.class_loss == 'focal':
            #     # calculate the class weights
            #     num_classes = class_logits.size(1)
            #     class_weights = compute_class_weights(true_class, num_classes)
            #     # apply Focal Loss
            #     criterion = FocalLoss(gamma=2.0, class_weights=class_weights)
            #     class_loss = criterion(class_logits, true_class)
            # elif args.class_loss == 'bsmCE':
            #     ## calculate the number of samples per class
            #     sample_per_class = []
            #     for i in range(args.num_classes):
            #         sample_per_class.append(torch.sum(true_class == i).item())
            #     # apply Balanced Softmax CE
            #     criterion = BalancedSoftmax(sample_per_class)
            #     class_loss = criterion(class_logits, true_class)

            if args.class_loss == 'cross_entropy':
                criterion = torch.nn.CrossEntropyLoss()
                class_loss = criterion(class_logits, true_class)
            else:
                raise ValueError(f"Loss function {args.class_loss} not supported!")
                
            class_loss.backward()
            optimizer_classifier.step()
            
            # update the statistics of classification results
            _class_pred = predict_class.detach().to('cpu', non_blocking=True)
            _class_true = true_class.detach().to('cpu', non_blocking=True)
            logger.update_stats(
                true=_class_true, 
                pred=_class_pred, 
                batch_size=_class_true.squeeze().size(0), 
                loss=class_loss if isinstance(class_loss, float) else class_loss.detach().cpu().item()
            )

        print(f"\n===== Epoch {epoch}/{args.epochs} - Elapsed: {time.time() - epoch_start_time:.2f}s =====")
        print("Classification results:")
        logger.write_epoch(split='train')
        
        ## ========== validation ========== ##
        val_class_res = eval_epoch(
            args, val_loader,
            classifier, device, split='val', criterion=criterion
        )
        #visualize_tsne(classifier, val_loader, device, num_samples=2000)
        
        if scheduler is not None:
            scheduler.step()
            
        ## ========== testing on other datasets ========== ##
        test_class_results = {}           
        eval_flag = False

        # if the f1 of the current classifier model is the highest, save the best model
        if val_class_res['f1'] > best_f1:
            best_f1 = val_class_res['f1']
            eval_flag = True
        
        if eval_flag :
            for test_name in test_loaders.keys():
                print(test_name)
                test_class_res = eval_epoch(
                    args, test_loaders[test_name], 
                    classifier, device, split='test', criterion=criterion
                )
                test_class_results[test_name] = test_class_res
        
        print( "=====================================")
        print(f" Best epoch: {epoch}, f1: {val_class_res['f1']}")
        print(f" Test results: {[res for res in test_class_results.values()]}")
        print( "=====================================")

#下游训练模型，即按照任务类别调用不同的训练模块
def downstream_train(args, dataset, device, cl_embeds=None):
    """ downstream task training for link prediction
    Args:
        args (argparse.Namespace): The arguments
        dataset (torch_geometric.data.InMemoryDataset): The dataset
        batch_index (torch.tensor): The batch index for all_node_embeds
        all_node_embeds (torch.tensor): The node embeddings come from the contrastive learning
        device (torch.device): The device to train the model on
    """
    # 删除以下代码：
    # if args.sgrl:
    #     dataset.set_cl_embeds(cl_embeds)
    
    # dataset.norm_nfeat([NET, DEV])#归一化节点特征
    # print(f"dataset.norm_nfeat([NET, DEV]):{dataset.norm_nfeat([NET, DEV])}")
    # assert 0

    
    
    # Subgraph sampling for each dataset graph & PE calculation
    (
        train_loader, val_loader, test_loaders, max_label
    ) = dataset_sampling(args, dataset) #获得训练集、验证集、测试集loader，最大标签数


    if args.task == 'regression':
        # set the loss function for regression
        # if args.regress_loss == 'gai':
        #     gmm_path = train_gmm(dataset)
        #     criterion = GAILoss(init_noise_sigma=args.noise_sigma, gmm=gmm_path, device=device)
        # elif args.regress_loss == 'bmc':
        #     criterion = BMCLoss(init_noise_sigma=args.noise_sigma, device=device)
        # elif args.regress_loss == 'bni':
        #     _, bin_edges, bin_count = get_lds_weights(
        #         dataset._data.edge_label[:, 1], 
        #         args.lds_kernel, args.lds_ks, args.lds_sigma
        #     )
        #     criterion = BNILoss(args.noise_sigma, bin_edges, bin_count,  device=device)
        # elif args.regress_loss == 'mse':
        #     criterion = torch.nn.MSELoss(reduction='mean')
        # elif args.regress_loss == 'lds':
        #     weights, _, _ = get_lds_weights(
        #         dataset._data.edge_label[:, 1], 
        #         args.lds_kernel, args.lds_ks, args.lds_sigma
        #     )
        #     dataset._data.edge_label[:, 1] = weights
        #     criterion = WeightedMSE()
        # else:
        #     raise ValueError(f"Loss func {args.regress_loss} not supported!")
        criterion = torch.nn.MSELoss(reduction='mean') #定义回归任务的损失函数
        
        
        start = time.time()
        model = GraphHead(args) #加载模型
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr) #定义优化器
        
        # 1) 每过 30 个 epoch，把 lr 降为原来的一半
        #动态调整学习率
        scheduler = StepLR(optimizer, step_size=40, gamma=0.5)
        # 或者，用监控 val loss 的方式：
        # scheduler = ReduceLROnPlateau(optimizer, mode='min',
        #                               factor=0.5, patience=5,
        #                               min_lr=1e-6, verbose=True)

        regress_train(args, model, optimizer, criterion,
                      train_loader, val_loader, test_loaders, max_label,
                      device, scheduler=scheduler)
        
    elif args.task == 'classification':
        model = GraphHead(args)
        start = time.time()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=40, gamma=0.5)
        class_train(args, model, optimizer, train_loader, val_loader, test_loaders, max_label,
                    device, scheduler=scheduler)
    
    else:
        raise ValueError(f"Task type {args.task} not supported!")
        

    elapsed = time.time() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print(f"Done! Training took {timestr}")
   






    
    
    

