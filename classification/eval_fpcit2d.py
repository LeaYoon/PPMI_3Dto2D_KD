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
                            resnet101, resnet152,\
                            MiddleLayer)
from models.DenseNet2D import DenseNetWrapper, densenet121, densenet169, densenet201, densenet161



# Teacher Model
from vit_pytorch.vit_3d import ViT as T_ViT
from models.CNN3D import PDNet as T_PDNet
from models.ResNet3D import ResNetWrapper as T_ResNetWrapper
from models.ResNet3D import resnet18 as resnet18_3d
from models.ResNet3D import resnet34 as resnet34_3d
from models.ResNet3D import resnet50 as resnet50_3d
from models.ResNet3D import resnet101 as resnet101_3d
from models.ResNet3D import resnet152 as resnet152_3d

# from data.fpcit2d import FPCIT2DDataset
# from data.fpcit3d import FPCIT3DDataset
from data.fpcit_integrated import FPCITIntegratedDataset
from data.av133_integrated import AV133IntegratedDataset

from data.DataIO import load_npz
from opt.Interpolation3D import ImageInterpolator
from opt.preprocess import to_categorical
from evaluation import get_performances, plot_trend, save_pred_label
from opt.losses import CELoss, L2Loss, rel_based_KDLoss, kNN_affinity, rbf_affinity, rbf_talor_affinity # , MetricLoss
from opt.lr_scheduler import CosineAnnealingWarmUpRestarts

from opt.graph_interpretation import save_graph_diff

    
def load_ckp(ckp_path, model):
    map_location = torch.device('cpu')
    ckp = torch.load(ckp_path, map_location=map_location)
    model_dict = model.state_dict()
    try:
        ckp_dict = ckp['model']
    except:
        ckp_dict = ckp
    pretrained_dict = {k: v for k, v in ckp_dict.items() if k in model_dict}
    unused_param = [k for k, v in ckp_dict.items() if k not in model_dict]
    lost_param = [k for k, v in model_dict.items() if k not in ckp_dict]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    print('<Visual> load teacher checkpoint from:', ckp_path)
    print('<Visual> unused param:', unused_param)
    print('<Visual> lost param:', lost_param)

    return model

def evaluate_fpcit2d(cfg):
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
    comp_method=cfg.INPUT.COMP_METHOD # "entropy" "rank"

    # if comp_method == "rank" and cfg.INPUT.JOINT_FUSION:
    #     joint_fusion = True
    #     in_channels=1
    # elif comp_method == "2D+e" and cfg.INPUT.JOINT_FUSION:
    #     early_fusion = True
    #     joint_fusion = True
    #     in_channels=1
    if cfg.INPUT.JOINT_FUSION:
        early_fusion=True
        joint_fusion=True
        in_channels=1

    view = cfg.INPUT.VIEW #"coronal" "axial" "saggital"
    channel_padding= cfg.INPUT.CHANNEL_PADDING # "copy", "zero" when early_fusion==False
    only_striatum = cfg.INPUT.ONLY_STRIATUM
    
    if cfg.TEST.DATASET=="fpcit":
        test_dataset = FPCITIntegratedDataset(masking=masking, early_fusion=early_fusion, view=view, comp_method=comp_method, channel_padding=channel_padding, _set="test", seed=11, resize=cfg.INPUT.RESIZE,only_striatum=only_striatum)
    elif cfg.TEST.DATASET=="av133":
        test_dataset = AV133IntegratedDataset(early_fusion=early_fusion, view=view, comp_method=comp_method, channel_padding=channel_padding, resize=cfg.INPUT.RESIZE, only_striatum=only_striatum)
    
    batch_size = cfg.TEST.BATCH_SIZE
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    # Model Load
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
    
    print("[!] in_channels", in_channels, view, early_fusion, joint_fusion)
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
        elif cfg.MODEL.BACKBONE.lower()=="densenet169":
            backbone = densenet169 
        
        model = DenseNetWrapper(backbone, output_dims=cfg.NUM_CLASSES, 
                                in_channels=in_channels, model_name = cfg.MODEL.BACKBONE.lower(), \
                                joint_fusion=joint_fusion)
    
    # remove last softmax layer
    # model = nn.Sequential(*list(model.children())[:-1])
    # model = model.to(device)

    # model = load_ckp(cfg.TEST.MODEL_PATH, model)
    ckp = torch.load(cfg.TEST.MODEL_PATH)
    # update backbone state_dict
    backbone_state_dict = {k.replace('backbone.', ''): v for k, v in ckp.items() if k.startswith('backbone.')}
    model.backbone.load_state_dict(backbone_state_dict)
    # update midfeat state_dict
    midfeat_state_dict = {k.replace('mid_feat.', ''): v for k, v in ckp.items() if k.startswith('mid_feat.')}
    model.mid_feat.load_state_dict(midfeat_state_dict)

    # model = model.to(device)
    model = model.cuda()
    
    ### Evaluating Model ###
    os.makedirs(cfg.EXP_ROOT, exist_ok=True)
    if not cfg.REP_EXEC:
        os.makedirs(os.path.join(cfg.EXP_ROOT, cfg.EXP_NAME), exist_ok=True)
        D_EXPERIMENTAL_DIR = os.path.join(cfg.EXP_ROOT, cfg.EXP_NAME)
    else:
        D_EXPERIMENTAL_DIR = os.path.join(cfg.EXP_ROOT)

    print("[!] D_EXPERIMENTAL_DIR", D_EXPERIMENTAL_DIR)
    print("[!] cfg.EXP_ROOT", cfg.EXP_ROOT)
    

    # again no gradients needed
    with torch.no_grad():
        model.eval()
        preds = []  
        onehot_labels = []
        filenames = []
        s_feats_list = []
        t_feats_list = []
        for data in test_loader:
            X_test, X_test_kd, y_test, filename = data
            X_test = X_test.to(device).float()
            X_test_kd = X_test_kd.to(device).float()

            print("[!] X_test", X_test.shape)    
            outputs, s_feats = model(X_test)

            if cfg.KD.TEACHER_MODEL:
                _model = copy.deepcopy(teacher)
                net = MiddleLayer(_model, 'avgpool')
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

