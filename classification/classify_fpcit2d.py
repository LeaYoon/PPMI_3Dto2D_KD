import os
import sys
import math
import copy

import numpy as np
from scipy import stats
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

import pydicom
from nibabel import load as nib_load # NiFti

### preprocess modules ###
from easydict import EasyDict
import torch
from torchsummary import summary
from sklearn.metrics import recall_score, classification_report, confusion_matrix

import torchvision
from torch.utils import data
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models

from vit_pytorch import ViT
from torchsummary import summary

from models.CNN2D import SimpleCNN
from models.ViT import ViTWrapper
from models.ResNet2D import (ResNetWrapper, \
                             resnet18, resnet34, resnet50, \
                            resnet101, resnet152)

from models.ResNet2D import MiddleLayer as MiddleLayer_ResNet


# Teacher Model
from vit_pytorch.vit_3d import ViT as T_ViT
from models.CNN3D import PDNet as T_PDNet
from models.ResNet3D import ResNetWrapper as T_ResNetWrapper
from models.ResNet3D import resnet18 as resnet18_3d
from models.ResNet3D import resnet34 as resnet34_3d
from models.ResNet3D import resnet50 as resnet50_3d
from models.ResNet3D import resnet101 as resnet101_3d
from models.ResNet3D import resnet152 as resnet152_3d
from models.DenseNet3D import DenseNetWrapper as T_DenseNetWrapper
from models.DenseNet3D import densenet121 as densenet121_3d
from models.DenseNet3D import densenet169 as densenet169_3d
from models.DenseNet3D import densenet201 as densenet201_3d

from models.DenseNet2D import (DenseNetWrapper, \
                             densenet121, densenet161)
from models.DenseNet2D import MiddleLayer as MiddleLayer_DenseNet

# from data.fpcit2d import FPCIT2DDataset
# from data.fpcit3d import FPCIT3DDataset
from data.fpcit_integrated import FPCITIntegratedDataset
from data.DataIO import load_npz
from opt.Interpolation3D import ImageInterpolator
from opt.preprocess import to_categorical
from evaluation import get_performances, plot_trend, save_pred_label
from opt.losses import CELoss, L2Loss, rel_based_KDLoss, kNN_affinity, rbf_affinity, rbf_talor_affinity # , MetricLoss
from opt.lr_scheduler import CosineAnnealingWarmUpRestarts

from opt.graph_interpretation import save_graph_diff

class MetricLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.L2Loss = L2Loss()
    
    def forward(self, X, sim_func=None):
        if sim_func is None:
            sim_func = torch.mm

        X = X / torch.norm(X, dim=1, keepdim=True)
        sim_mat = sim_func(X, X.transpose(0, 1))
        sim_mat = sim_mat / torch.max(sim_mat)

        identity = torch.eye(X.size(0)).to(X.device)
        return self.L2Loss(sim_mat, identity)
    
def load_ckp(ckp_path, model):
    map_location = torch.device('cpu')
    ckp = torch.load(ckp_path, map_location=map_location)
    model_dict = model.state_dict()
    ckp_dict = ckp['model']
    pretrained_dict = {k: v for k, v in ckp_dict.items() if k in model_dict}
    unused_param = [k for k, v in ckp_dict.items() if k not in model_dict]
    lost_param = [k for k, v in model_dict.items() if k not in ckp_dict]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    print('<Visual> load teacher checkpoint from:', ckp_path)
    print('<Visual> unused param:', unused_param)
    print('<Visual> lost param:', lost_param)

    return model

def classify_fpcit2d(cfg):
    device = torch.device('cuda' if torch.cuda.is_available () else 'cpu')
    print("device", device)

    # check input integrity
    if cfg.INPUT.EARLY_FUSION and cfg.INPUT.CHANNEL_SIZE==1:
        # assert False, "[!] False in 'cfg.INPUT.EARLY_FUSION and cfg.INPUT.CHANNEL_SIZE==1', check the arguments!!"
        cfg.INPUT.CHANNEL_SIZE=3
    elif not cfg.INPUT.EARLY_FUSION and cfg.INPUT.CHANNEL_SIZE==3:
        cfg.INPUT.CHANNEL_SIZE=1
    in_channels = cfg.INPUT.CHANNEL_SIZE
    masking=cfg.INPUT.MASKING
    early_fusion=cfg.INPUT.EARLY_FUSION
    joint_fusion = cfg.INPUT.JOINT_FUSION
    if cfg.INPUT.JOINT_FUSION:
        early_fusion = True
        joint_fusion = True
        in_channels=1

    comp_method=cfg.INPUT.COMP_METHOD # "entropy" "rank"
    view = cfg.INPUT.VIEW #"coronal" "axial" "saggital"
    channel_padding= cfg.INPUT.CHANNEL_PADDING # "copy", "zero" when early_fusion==False
    only_striatum = cfg.INPUT.ONLY_STRIATUM

    train_dataset = FPCITIntegratedDataset(masking=masking, early_fusion=early_fusion, view=view, comp_method=comp_method, channel_padding=channel_padding, _set="train", seed=11, resize=cfg.INPUT.RESIZE,only_striatum=only_striatum)
    val_dataset = FPCITIntegratedDataset(masking=masking, early_fusion=early_fusion, view=view, comp_method=comp_method, channel_padding=channel_padding, _set="val", seed=11, resize=cfg.INPUT.RESIZE,only_striatum=only_striatum)
    test_dataset = FPCITIntegratedDataset(masking=masking, early_fusion=early_fusion, view=view, comp_method=comp_method, channel_padding=channel_padding, _set="test", seed=11, resize=cfg.INPUT.RESIZE,only_striatum=only_striatum)
    # train_dataset = FPCIT2DDataset(masking=masking, early_fusion=early_fusion, view=view, comp_method=comp_method, channel_padding=channel_padding, _set="train", seed=11, resize=cfg.INPUT.RESIZE,only_striatum=only_striatum)
    # val_dataset = FPCIT2DDataset(masking=masking, early_fusion=early_fusion, view=view, comp_method=comp_method, channel_padding=channel_padding, _set="val", seed=11, resize=cfg.INPUT.RESIZE, only_striatum=only_striatum)
    # test_dataset = FPCIT2DDataset(masking=masking, early_fusion=early_fusion, view=view, comp_method=comp_method, channel_padding=channel_padding, _set="test", seed=11, resize=cfg.INPUT.RESIZE, only_striatum=only_striatum)

    batch_size = cfg.TRAIN.BATCH_SIZE
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    # Data for Teacher Model
    # if cfg.KD.TEACHER_MODEL:
            
    #     # train_kd_dataset = FPCIT3DDataset(masking=masking, _set="train", seed=11, resize = cfg.INPUT.RESIZE, ch_padding=channel_padding, only_striatum=only_striatum)
    #     # val_kd_dataset = FPCIT3DDataset(masking=masking, _set="val", seed=11, resize = cfg.INPUT.RESIZE, ch_padding=channel_padding, only_striatum=only_striatum)

    #     batch_size = cfg.TRAIN.BATCH_SIZE
    #     train_kd_loader = torch.utils.data.DataLoader(train_kd_dataset, batch_size=batch_size,
    #                                             shuffle=False, num_workers=0)
    #     val_kd_loader = torch.utils.data.DataLoader(val_kd_dataset, batch_size=batch_size,
    #                                             shuffle=False, num_workers=0)
    
    # if channel_padding:
    #     in_channels = 3
    # else:
    #     in_channels = 1

    # if channel_padding is None:
    #     in_channels = cfg.INPUT.CHANNEL_SIZE

    # Teacher Model Load
    if cfg.KD.TEACHER_MODEL:
        print(type(cfg.KD.TEACHER_MODEL))
        print(cfg.KD.TEACHER_MODEL)
        model_name = os.path.basename(cfg.KD.TEACHER_MODEL)
        model_name = model_name.split("_")[0]

        if model_name=="PDNet":
            teacher = T_PDNet(in_channels)
        elif model_name=="ViT3D":
            teacher = T_ViT(image_size=cfg.INPUT.RESIZE[1],
            frames = cfg.INPUT.RESIZE[0],
            image_patch_size = cfg.MODEL.VIT_PATCH_SIZE, # 8*8 grid
            frame_patch_size= cfg.MODEL.VIT_PATCH_SIZE, # 8
            num_classes=2,
            dim=cfg.MODEL.VIT_DIM, # 128
            depth=cfg.MODEL.VIT_DEPTH, # 6
            heads=cfg.MODEL.VIT_HEADS, # 16
            mlp_dim=cfg.MODEL.VIT_MLP_DIM, # 32
            channels=in_channels) # 128
            
            teacher = ViTWrapper(teacher)
        elif "resnet" in model_name or "ResNet" in model_name:
            if model_name[-2:]=="18":
                backbone_kd = resnet18_3d 
            elif model_name[-2:]=="34":
                backbone_kd = resnet34_3d  
            elif model_name[-2:]=="50":
                backbone_kd = resnet50_3d  
            elif model_name[-3:]=="101":
                backbone_kd = resnet101_3d  
            elif model_name[-3:]=="152":
                backbone_kd = resnet152_3d  
            teacher = T_ResNetWrapper(backbone_kd, cfg.NUM_CLASSES, model_name = model_name.lower(), in_channels=1)
        elif "densenet" in model_name or "DenseNet" in model_name:
            if model_name[-3:]=="121":
                backbone_kd = densenet121_3d
            elif model_name[-3:]=="169":
                backbone_kd = densenet169_3d
            elif model_name[-3:]=="201":
                backbone_kd = densenet201_3d
            teacher = T_DenseNetWrapper(backbone_kd, cfg.NUM_CLASSES, model_name = model_name.lower(), in_channels=1)

        ### Initiate Teacher ###
        teacher = load_ckp(cfg.KD.TEACHER_MODEL, teacher)

        # Frozen the Teacher
        for p in teacher.parameters():
            p.requires_grad = False
        
        # remove last softmax layer
        # teacher = nn.Sequential(*list(teacher.children())[:-1])
        teacher = getattr(teacher, 'backbone')
        teacher = teacher.to(device)
        teacher.eval()


    if comp_method == "2D+e" and (early_fusion ==True or joint_fusion==True):
        in_channels = 9
    elif comp_method == "2D+e" and (early_fusion ==False and joint_fusion==False):
        in_channels = 3
    elif comp_method != "2D+e" and (early_fusion ==True or joint_fusion==True):
        in_channels = 3
    elif comp_method != "2D+e" and (early_fusion ==False and joint_fusion==False):
        in_channels = 1
    else:
        raise NotImplemented
    
    # print("[!] in_channels", in_channels, view, early_fusion, joint_fusion)
    ### Student Model Load ###
    if cfg.MODEL.BACKBONE=="SimpleCNN":
        model = SimpleCNN(in_channels)
        
    elif cfg.MODEL.BACKBONE=="ViT2D":
        model = ViT(image_size=cfg.INPUT.RESIZE[1],
                    patch_size = cfg.MODEL.VIT_PATCH_SIZE, # 8*8 grid
                    num_classes=2,
                    dim=cfg.MODEL.VIT_DIM,
                    depth=cfg.MODEL.VIT_DEPTH,
                    heads=cfg.MODEL.VIT_HEADS,
                    mlp_dim=cfg.MODEL.VIT_MLP_DIM,
                    channels=in_channels)

        model = ViTWrapper(model)
    elif "resnet" in cfg.MODEL.BACKBONE or "resnet" in cfg.MODEL.BACKBONE.lower():
        if cfg.MODEL.BACKBONE.lower()=="resnet18":
            backbone = resnet18 
        elif cfg.MODEL.BACKBONE.lower()=="resnet34":
            backbone = resnet34 
        elif cfg.MODEL.BACKBONE.lower()=="resnet50":
            backbone = resnet50 
        elif cfg.MODEL.BACKBONE.lower()=="resnet101":
            backbone = resnet101 
        elif cfg.MODEL.BACKBONE.lower()=="resnet152":
            backbone = resnet152 
        model = ResNetWrapper(backbone, output_dims=cfg.NUM_CLASSES, pretrained=cfg.MODEL.PRETRAINED_RESNET, \
                              in_channels=in_channels, model_name = cfg.MODEL.BACKBONE.lower(), \
                              joint_fusion=joint_fusion)
    elif "densenet" in cfg.MODEL.BACKBONE or "densenet" in cfg.MODEL.BACKBONE.lower():
        if cfg.MODEL.BACKBONE.lower()=="densenet121":
            backbone = densenet121 
        elif cfg.MODEL.BACKBONE.lower()=="densenet161":
            backbone = densenet161 
        model = DenseNetWrapper(backbone, output_dims=cfg.NUM_CLASSES, \
                              in_channels=in_channels, model_name = cfg.MODEL.BACKBONE.lower(), \
                              joint_fusion=joint_fusion)

    # remove last softmax layer
    # model = nn.Sequential(*list(model.children())[:-1])
    # model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # KD Loss
    metric_loss = MetricLoss()
    if cfg.KD.TEACHER_MODEL:
        if cfg.KD.LOSS=="peng":
            kd_loss = rel_based_KDLoss()    
            sim_func = None
            # l2_loss = L2Loss()
            kl_loss = nn.KLDivLoss(log_target=True)
        elif cfg.KD.LOSS == "CE":
            # kd_loss = CELoss()
            kd_loss = nn.KLDivLoss(log_target=True)
        elif cfg.KD.LOSS == "L2":
            kd_loss = L2Loss()
        elif cfg.KD.LOSS == "GS_f": # Graph Similarity Loss using logit and final features
            kd_loss = rel_based_KDLoss()    
            sim_func = None
        elif cfg.KD.LOSS == "GS_lf": # Graph Similarity Loss using logit and final features
            kd_loss = rel_based_KDLoss()    
            sim_func = None
        # elif cfg.KD.LOSS == "GS_lf_rbf": # Graph Similarity Loss using logit and final features
        #     kd_loss = rel_based_KDLoss()    
        #     rbf_sim = rbf_affinity(sigma=1, knn=5)
        #     sim_func = rbf_sim
        # elif cfg.KD.LOSS == "GS_lf_rbf_taylor": # Graph Similarity Loss using logit and final features
        #     kd_loss = rel_based_KDLoss()    
        #     rbf_sim_taylor = rbf_talor_affinity(delta=1, P=3)
        #     sim_func = rbf_sim_taylor
        # elif cfg.KD.LOSS == "GS_lf_knn": # Graph Similarity Loss using logit and final features
        #     kd_loss = rel_based_KDLoss()    
        #     knn_sim=kNN_affinity(knn=5)
        #     sim_func = knn_sim
            
    # lr_rate = cfg.TRAIN.LEARNING_RATE
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate) # 0.00005 # 0.001
    optimizer = torch.optim.RAdam(model.parameters(), lr=0) # 0.00005 # 0.001
    # optimizer = torch.optim.RAdam(model.parameters(), lr=lr_rate) # 0.00005 # 0.001
    # scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=1, eta_min=0.000001)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer=optimizer, T_0=20, T_mult=1, T_up=10, eta_max=0.01, gamma=0.5)
    # summary(v, (3, cfg.resize[0], cfg.resize[1],cfg.resize[2])) #  >_<!

    ### Training Model ###
    ### Experimental param
    os.makedirs(cfg.EXP_ROOT, exist_ok=True)
    if not cfg.REP_EXEC:
        os.makedirs(os.path.join(cfg.EXP_ROOT, cfg.EXP_NAME), exist_ok=True)
        D_EXPERIMENTAL_DIR = os.path.join(cfg.EXP_ROOT, cfg.EXP_NAME)
    else:
        D_EXPERIMENTAL_DIR = os.path.join(cfg.EXP_ROOT)

    num_epochs = cfg.TRAIN.NUM_EPOCH #500 #
    print_step = cfg.TRAIN.PRINT_STEP # 10 # ?
    patience_limit = cfg.TRAIN.PATIENCE_LIMIT # 20

    best_loss=cfg.TRAIN.BEST_LOSS_INIT
    train_history = []
    val_history = []
    train_kd_history = []
    val_kd_history = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()

        # if cfg.KD.TEACHER_MODEL:
        #     train_kd_loader_iter = iter(train_kd_loader)

        train_loss = 0.0
        val_loss = 0.0
        train_kd_loss = 0.0
        val_kd_loss = 0.0
        for i, data in enumerate(train_loader):
            
            # get the inputs; data is a list of [inputs, labels, filenames]
            X_train, X_train_kd, y_train, filenames = data
            
            ### Model Input Check ###
            # import matplotlib.pyplot as plt
            # test_path = r"/home/project/experiments"
            # test_x= X_train[0].permute(1,2,0).numpy()
            # print("[!!!!!!!!!!!]", test_x.shape) # 텐서플로말고 파이토치에서는 (3, h, w) 식이라, permute 해야함.
            # plt.imshow(test_x)
            # plt.colorbar()
            # plt.savefig(os.path.join(test_path, cfg.EXP_NAME+".png")) # exp_name 으로 파일 이름 분기 해서 저장
            # exit() #  여기서 프로그램 종료
            ### ###

            X_train = X_train.to(device).float()
            
            model = model.cuda()
            outputs, s_feats = model(X_train) # (64, 1, 64, 64)
            
            
            cls_loss=0.0
            train_loss_kd=0.0
            total_loss = 0.0
            if not cfg.KD.ONLY_KD:
                # CLS loss
                
                cls_loss = criterion(outputs, y_train.cuda().long())
                if not torch.isfinite(cls_loss):
                    print('WARNING: non-finite loss, ending training,',cls_loss)
                    exit(1)
                
                # zero the parameter gradients
                # optimizer.zero_grad()
                # cls_loss.backward()
                total_loss+=cls_loss*cfg.TRAIN.CLS_LOSS_WEIGHT
                # optimizer.step()

            # print statistics
            try:
                train_loss += cls_loss.item()
            except AttributeError:
                train_loss += cls_loss

            ### Knowledge Distillation & Dillation Loss ###
            if cfg.KD.TEACHER_MODEL:
                # X_train_kd, _, _ = next(train_kd_loader_iter)
                X_train_kd = X_train_kd.to(device).float()
                t_outputs = teacher(X_train_kd)

                soft_targets=None
                soft_prob = None
                soft_targets = nn.functional.log_softmax(t_outputs / cfg.KD.TEMPERATURE, dim=-1)
                soft_prob = nn.functional.log_softmax(outputs / cfg.KD.TEMPERATURE, dim=-1)
                if cfg.KD.LOSS == "peng":
                    # extract teacher feature
                    _model = copy.deepcopy(teacher)
                    if "resnet" in cfg.MODEL.BACKBONE.lower():
                        net = MiddleLayer_ResNet(_model, 'avgpool')
                    elif "densenet" in cfg.MODEL.BACKBONE.lower():
                        net = MiddleLayer_DenseNet(_model, 'avgpool')
                    t_feats = net(X_train_kd).flatten(start_dim=1)
                    del _model
                    # rel-based KD using feature graph
                    _train_loss_kd = kd_loss(s_feats, t_feats, sim_func = sim_func)*cfg.KD.LOSS_RATIO
                    total_loss+=_train_loss_kd

                    # response-based KD
                    train_loss_kd = kl_loss(soft_prob, soft_targets)*cfg.KD.LOSS_RATIO
                    
                    if not torch.isfinite(train_loss_kd):
                        print('WARNING: non-finite loss, ending training,', train_loss_kd)
                        exit(1)

                    total_loss+=train_loss_kd
                    train_loss_kd+=_train_loss_kd
                elif cfg.KD.LOSS == "GS_f":
                    # extract teacher feature
                    _model = copy.deepcopy(teacher)
                    if "resnet" in cfg.MODEL.BACKBONE.lower():
                        net = MiddleLayer_ResNet(_model, 'avgpool')
                    elif "densenet" in cfg.MODEL.BACKBONE.lower():    
                        net = MiddleLayer_DenseNet(_model, 'avgpool')

                    t_feats = net(X_train_kd).flatten(start_dim=1)
                    del _model
                    # rel-based KD using feature graph
                    _train_loss_kd = kd_loss(s_feats, t_feats, sim_func = sim_func)*cfg.KD.LOSS_RATIO
                    total_loss+=_train_loss_kd

                    train_loss_kd+=_train_loss_kd
                elif cfg.KD.LOSS == "GS_lf" or cfg.KD.LOSS == "GS_lf_knn" or cfg.KD.LOSS == "GS_lf_rbf" or cfg.KD.LOSS=="GS_lf_rbf_taylor":
                    # extract teacher feature
                    _model = copy.deepcopy(teacher)
                    if "resnet" in cfg.MODEL.BACKBONE.lower():
                        net = MiddleLayer_ResNet(_model, 'avgpool')
                    elif "densenet" in cfg.MODEL.BACKBONE.lower():    
                        net = MiddleLayer_DenseNet(_model, 'avgpool')
                    t_feats = net(X_train_kd).flatten(start_dim=1)
                    del _model
                    # rel-based KD using feature graph
                    _train_loss_kd = kd_loss(s_feats, t_feats, sim_func = sim_func)*cfg.KD.LOSS_RATIO
                    total_loss+=_train_loss_kd

                    # rel-based KD using logit graph
                    _train_loss_kd = kd_loss(soft_prob, soft_targets, sim_func = sim_func)*cfg.KD.LOSS_RATIO
                    total_loss+=_train_loss_kd

                    train_loss_kd+=_train_loss_kd

                    # metric loss
                    _train_loss_metric = metric_loss(s_feats)
                    total_loss+= _train_loss_metric
                    train_loss_kd+=_train_loss_metric
                    _train_loss_metric = metric_loss(t_feats)
                    total_loss+= _train_loss_metric
                    train_loss_kd+=_train_loss_metric
                    
                else:
                    train_loss_kd = kd_loss(soft_prob, soft_targets)*cfg.KD.LOSS_RATIO
                
                    if not torch.isfinite(train_loss_kd):
                        print('WARNING: non-finite loss, ending training,', train_loss_kd)
                        exit(1)

                    total_loss+=train_loss_kd
                
            # print statistics
            # train_kd_loss += train_loss_kd.item()
            try:
                train_kd_loss += train_loss_kd.item()
            except AttributeError:
                train_kd_loss += train_loss_kd
                
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        train_loss =train_loss/X_train.size(0)
        train_history.append(train_loss)   
        
        if cfg.KD.TEACHER_MODEL:
            train_kd_loss = train_kd_loss/X_train.size(0)
            train_kd_history.append(train_kd_loss)    

        # again no gradients needed
        with torch.no_grad():
            # if cfg.KD.TEACHER_MODEL:
                # val_kd_loader_iter = iter(val_kd_loader)
            preds = []  
            for j, data in enumerate(val_loader):
                X_val, X_val_kd, y_val, filename = data
                X_val = X_val.to(device).float()
                
                outputs, s_feats = model(X_val)
                _, predictions = torch.max(outputs.data, 1) # returns (max_values, indice)
                loss = criterion(outputs, y_val.cuda())
                val_loss+=loss.item()*cfg.TRAIN.CLS_LOSS_WEIGHT
                preds.append(predictions.data.cpu().numpy())

                if cfg.KD.TEACHER_MODEL:
                    # X_val_kd, _, _ = next(val_kd_loader_iter)
                    X_val_kd = X_val_kd.to(device).float()
                    t_outputs = teacher(X_val_kd)

                    soft_targets=None
                    soft_prob = None
                    soft_targets = nn.functional.log_softmax(t_outputs / cfg.KD.TEMPERATURE, dim=-1)
                    soft_prob = nn.functional.log_softmax(outputs / cfg.KD.TEMPERATURE, dim=-1)

                    # extract teacher feature
                    if cfg.KD.LOSS == "peng":
                        # extract teacher feature
                        _model = copy.deepcopy(teacher)
                        if "resnet" in cfg.MODEL.BACKBONE.lower():
                            net = MiddleLayer_ResNet(_model, 'avgpool')
                        elif "densenet" in cfg.MODEL.BACKBONE.lower():    
                            net = MiddleLayer_DenseNet(_model, 'avgpool')
                        t_feats = net(X_val_kd).flatten(start_dim=1)
                        del _model
                        # rel-based KD using feature graph
                        _val_loss_kd = kd_loss(s_feats, t_feats, sim_func = sim_func)*cfg.KD.LOSS_RATIO
                        val_kd_loss+=_val_loss_kd.item()

                        # response-based KD
                        val_loss_kd = kl_loss(soft_prob, soft_targets)*cfg.KD.LOSS_RATIO
                        val_kd_loss+=val_loss_kd.item()

                        val_loss_kd+=_val_loss_kd
                    elif cfg.KD.LOSS == "GS_f":
                        # extract teacher feature
                        _model = copy.deepcopy(teacher)
                        if "resnet" in cfg.MODEL.BACKBONE.lower():
                            net = MiddleLayer_ResNet(_model, 'avgpool')
                        elif "densenet" in cfg.MODEL.BACKBONE.lower():    
                            net = MiddleLayer_DenseNet(_model, 'avgpool')
                        t_feats = net(X_val_kd).flatten(start_dim=1)
                        del _model

                        # rel-based KD using feature graph
                        _val_loss_kd = kd_loss(s_feats, t_feats, sim_func = sim_func)*cfg.KD.LOSS_RATIO
                        val_kd_loss+=_val_loss_kd
                        val_loss_kd+=_val_loss_kd
                    elif cfg.KD.LOSS == "GS_lf" or cfg.KD.LOSS == "GS_lf_rbf" or cfg.KD.LOSS == "GS_lf_knn" or cfg.KD.LOSS=="GS_lf_rbf_taylor":
                        # extract teacher feature
                        _model = copy.deepcopy(teacher)
                        if "resnet" in cfg.MODEL.BACKBONE.lower():
                            net = MiddleLayer_ResNet(_model, 'avgpool')
                        elif "densenet" in cfg.MODEL.BACKBONE.lower():    
                            net = MiddleLayer_DenseNet(_model, 'avgpool')
                        t_feats = net(X_val_kd).flatten(start_dim=1)
                        del _model

                        # rel-based KD using feature graph
                        _val_loss_kd = kd_loss(s_feats, t_feats, sim_func = sim_func)*cfg.KD.LOSS_RATIO
                        val_kd_loss+=_val_loss_kd.item()

                        # rel-based KD using logit graph
                        val_loss_kd = kd_loss(soft_prob, soft_targets, sim_func = sim_func)*cfg.KD.LOSS_RATIO
                        val_kd_loss+=val_loss_kd.item()
                        val_loss_kd+=_val_loss_kd

                        # metric loss
                        _val_loss_metric = metric_loss(s_feats)
                        val_kd_loss+= _val_loss_metric.item()
                        val_loss_kd+=_val_loss_metric
                        _val_loss_metric = metric_loss(t_feats)
                        val_kd_loss+= _val_loss_metric.item()
                        val_loss_kd+=_val_loss_metric

                    else:
                        val_loss_kd = kd_loss(soft_prob, soft_targets)*cfg.KD.LOSS_RATIO
                        val_kd_loss+=val_loss_kd.item()

        val_loss = val_loss/X_val.size(0)
        val_history.append(val_loss)    
        
        if cfg.KD.TEACHER_MODEL:
            val_kd_loss =val_kd_loss/X_val.size(0)
            val_kd_history.append(val_kd_loss)  
        
        if epoch % print_step == 0:
            if cfg.KD.TEACHER_MODEL:
                print(f'[Epoch: {epoch + 1}] train loss: {train_history[-1]:.3f}, val loss: {val_history[-1]:.3f}, train kd loss: {train_kd_history[-1]:.3f}, val kd loss: {val_kd_history[-1]:.3f}')
            else:    
                print(f'[Epoch: {epoch + 1}] train loss: {train_history[-1]:.3f}, val loss: {val_history[-1]:.3f}')
        

        # early stopping check
        if (val_loss+val_kd_loss)/2 > best_loss: # loss is not improved
            patience_check += 1

            if patience_check >= patience_limit: 
                print("[!] Training is terminated by early stopping !!!")
                break

        else: # loss가 개선된 경우
            best_loss = (val_loss+val_kd_loss)/2
            patience_check = 0

            ### Model Save ###
            if cfg.MODEL.SAVE_MODEL :
                checkpoint_path = os.path.join(D_EXPERIMENTAL_DIR, "model")
                os.makedirs(checkpoint_path, exist_ok=True)

                # save_dict = {
                #     'model': model.state_dict(),
                #     'optimizer': optimizer.state_dict()
                # }
                # torch.save(save_dict, os.path.join(checkpoint_path, f"best_model.pth"))
                torch.save(model.state_dict(), os.path.join(checkpoint_path, f"best_model.pth"))

    print('Finished Training')
    if cfg.MODEL.SAVE_MODEL:
        best_state_dict = torch.load(os.path.join(checkpoint_path, f"best_model.pth"))
        model.load_state_dict(best_state_dict)
    model.eval()

    history = EasyDict()
    history.history=EasyDict()
    history.history["loss"] = train_history
    history.history["val_loss"] = val_history

    if cfg.KD.TEACHER_MODEL:
        kd_history = EasyDict()
        kd_history.history=EasyDict()
        kd_history.history["loss"] = train_kd_history
        kd_history.history["val_loss"] = val_kd_history

    
    # cfg.PROJECT_ROOT = project_root
    # cfg.EXP_ROOT = cfg.PROJECT_ROOT / 'experiments'
    # cfg.EXP_NAME = ''
    # cfg.DATA_DIRECTORY_SOURCE = cfg.PROJECT_ROOT / 'dataset'

#     # #'d:\\윤혜민\\[핵의학과]\\FPCCIT_Ex_3D_Image_Classification'
#     # main_dir = cfg.PROJECT_ROOT
#     # proj_dirname = cfg.proj_dirname
#     # ex_dirname = cfg.ex_dirname #"Ex_2D_cls_ViT_EarlyFusion_epoch500_v1.0"
    # if not os.path.isdir(cfg.EXP_ROOT) and not cfg.REP_EXEC:
    #     os.mkdir(cfg.EXP_ROOT)
    # if not os.path.isdir(os.path.join(cfg.EXP_ROOT, cfg.EXP_NAME)) and not cfg.REP_EXEC:
    #     os.mkdir(os.path.join(cfg.EXP_ROOT, cfg.EXP_NAME))
    
    plot_trend(history, save_path=os.path.join(D_EXPERIMENTAL_DIR, "learning_curve.png"))
    if cfg.KD.TEACHER_MODEL:
        plot_trend(kd_history, save_path=os.path.join(D_EXPERIMENTAL_DIR, "learning_curve_KD.png"))

    
    # again no gradients needed
    with torch.no_grad():
        preds = []  
        onehot_labels = []
        filenames = []
        s_feats_list = []
        t_feats_list = []
        for data in test_loader:
            X_test, X_test_kd, y_test, filename = data
            X_test = X_test.to(device).float()
            X_test_kd = X_test_kd.to(device).float()
            
            outputs, s_feats = model(X_test)

            if cfg.KD.TEACHER_MODEL:
                _model = copy.deepcopy(teacher)
                if "resnet" in cfg.MODEL.BACKBONE.lower():
                    net = MiddleLayer_ResNet(_model, 'avgpool')
                elif "densenet" in cfg.MODEL.BACKBONE.lower():    
                    net = MiddleLayer_DenseNet(_model, 'avgpool')
                t_feats = net(X_test_kd).flatten(start_dim=1)
                del _model

            preds.append(outputs.data.cpu().numpy())
            onehot_labels.append(y_test.data.cpu().numpy())
            filenames.append(filename)
            if cfg.KD.TEACHER_MODEL:
                s_feats_list.append(s_feats)
                t_feats_list.append(t_feats)
                
    # save graph log
    if cfg.KD.SAVE_TEST_GRAPH:
        s_feats = torch.cat(s_feats_list, dim=0)
        t_feats = torch.cat(t_feats_list, dim=0)
        save_graph_diff(D_EXPERIMENTAL_DIR, s_feats, t_feats, None)
                    
    preds = np.concatenate(preds, axis=0)
    onehot_labels = np.concatenate(onehot_labels, axis=0)
    filenames = np.concatenate(filenames, axis=0)


    print("[!] preds", type(preds), preds.shape)
    print("[!] onehot_labels", type(onehot_labels), onehot_labels.shape)
    print("[!] filenames", type(filenames))


    # pred_proba = np.array(pred).max(axis=1)  # (N, )
    onehot_labels = to_categorical(onehot_labels, 2)
    pred_ind_list = np.array(preds).argmax(axis=1)  # (N, )
    label_ind_list = np.array(onehot_labels).argmax(axis=1)  # (N, )
    
    # sample_based_analysis result
    save_pred_label(onehot_labels, preds, save_filepath=os.path.join(D_EXPERIMENTAL_DIR, "sample_based_analysis.xlsx"),
                    onehot_label = True, filename_list = filenames) # y_test_ : (N, Num_classes)

    conf_m = confusion_matrix(label_ind_list, pred_ind_list)
    print("CV #", "confusion matrix")
    print(conf_m)

    test_acc, test_macro_f1_score, test_micro_f1_score, test_weighted_f1_score, test_g_mean = get_performances(label_ind_list, pred_ind_list)
    print("Test ACC", test_acc)
    print("Test Macro F1", test_macro_f1_score)
    print("Test Micro F1", test_micro_f1_score)
    print("Test Weighted F1", test_weighted_f1_score)
    print("Test G_mean", test_g_mean)

    # logging the conf_m
    result_save_filename = os.path.join(D_EXPERIMENTAL_DIR,
                                        "performance_report.txt")
    with open(result_save_filename, "w") as f:
        f.write("Accuracy : " + str(test_acc) + '\n')
        f.write("Macro F1 : " + str(test_macro_f1_score)+'\n')
        f.write("Micro F1 : " + str(test_micro_f1_score)+'\n')
        f.write("Weighted F1 : " + str(test_weighted_f1_score)+'\n')
        f.write("Gmean F1 : " + str(test_g_mean)+'\n')
        for row in conf_m:
            f.write("%s\n" % row)

    target_names = ['HC','PD']
    result = classification_report(label_ind_list, pred_ind_list, target_names=target_names)
    print("Test Phase result")
    print(result)
    with open(result_save_filename, "a") as f:
        f.write("%s\n" % result)

    results = dict()
    for item in cfg.METRICS:
        if item=="ACCURACY":
            results["ACCURACY"]=test_acc    
        elif item=="RECALL":
            results["RECALL"]=recall_score(label_ind_list, pred_ind_list)     
        elif item=="WEIGHTED_F1":
            results["WEIGHTED_F1"]=test_weighted_f1_score    

    return results

