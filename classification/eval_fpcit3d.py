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

from models.CNN3D import PDNet
from models.ViT import ViTWrapper
from models.ResNet3D import ResNetWrapper, resnet18, resnet34, resnet50, resnet101, resnet152
from models.DenseNet3D import DenseNetWrapper, densenet121, densenet161, densenet201


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

def evaluate_fpcit3d(cfg):
    device = torch.device('cuda' if torch.cuda.is_available () else 'cpu')
    print("device", device)

    # check input integrity
    if cfg.INPUT.RESIZE[0]==64 and cfg.MODEL.BACKBONE=="PDNet":
        cfg.INPUT.RESIZE[0]=109
        cfg.INPUT.RESIZE[1]=109
        cfg.INPUT.RESIZE[2]=109
    # cfg.INPUT.RESIZE[0]=109
    # cfg.INPUT.RESIZE[1]=109
    # cfg.INPUT.RESIZE[2]=109
    fc_in=None
    # print("[!] cfg.INPUT.RESIZE", cfg.INPUT.RESIZE)

    masking=cfg.INPUT.MASKING
    # early_fusion=cfg.INPUT.EARLY_FUSION
    # comp_method=cfg.INPUT.COMP_METHOD # "entropy" "rank"
    view = cfg.INPUT.VIEW #"coronal" "axial" "saggital"
    channel_padding= cfg.INPUT.CHANNEL_PADDING # "copy", "zero" when early_fusion==False
    only_striatum = cfg.INPUT.ONLY_STRIATUM

    # ### transform ###
    # random_flip = tio.RandomFlip(axes=('LR',))
    # random_affine = tio.RandomAffine()
    # random_elastic_deformation = tio.RandomElasticDeformation()
    # blurring_transform = tio.RandomBlur(std=0.3)
    # random_noise = tio.RandomNoise(std=(0, 0.3))
    # random_gamma = tio.RandomGamma(log_gamma=(-0.5, 0.5))

    # transform = tio.Compose([random_flip,
    #                          random_affine,
    #                          random_elastic_deformation,
    #                          blurring_transform,
    #                          random_noise,
    #                         random_gamma])
    transform=None
    early_fusion = False
    comp_method="rank"
    
    if cfg.TEST.DATASET=="fpcit":
        test_dataset = FPCITIntegratedDataset(masking=masking, early_fusion=early_fusion, view=view, comp_method=comp_method, channel_padding=channel_padding, _set="test", seed=11, resize=cfg.INPUT.RESIZE,only_striatum=only_striatum)
    elif cfg.TEST.DATASET=="av133":
        test_dataset = AV133IntegratedDataset(early_fusion=early_fusion, view=view, comp_method=comp_method, channel_padding=channel_padding, resize=cfg.INPUT.RESIZE, only_striatum=only_striatum)
    
    batch_size = cfg.TEST.BATCH_SIZE
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    if channel_padding:
        in_channels = 3
    else:
        in_channels = 1

    # Model Load
    if cfg.MODEL.BACKBONE=="PDNet":
        model = PDNet(in_channels)
    elif cfg.MODEL.BACKBONE=="ViT3D":
        model = ViT(image_size=cfg.INPUT.RESIZE[1],
        frames = cfg.INPUT.RESIZE[0],
        image_patch_size = cfg.MODEL.VIT_PATCH_SIZE, # 8*8 grid
        frame_patch_size= cfg.MODEL.VIT_PATCH_SIZE, # 8
        num_classes=2,
        dim=cfg.MODEL.VIT_DIM, # 128
        depth=cfg.MODEL.VIT_DEPTH, # 6
        heads=cfg.MODEL.VIT_HEADS, # 16
        mlp_dim=cfg.MODEL.VIT_MLP_DIM, # 32
        channels=in_channels) # 128
        
        model = ViTWrapper(model)
    elif "resnet" in cfg.MODEL.BACKBONE.lower():
        if cfg.MODEL.BACKBONE.lower()=="resnet18":
            backbone = resnet18 
            fc_in = 512
        elif cfg.MODEL.BACKBONE.lower()=="resnet34":
            backbone = resnet34 
        elif cfg.MODEL.BACKBONE.lower()=="resnet50":
            backbone = resnet50 
        elif cfg.MODEL.BACKBON.lower()=="resnet101":
            backbone = resnet101 
        elif cfg.MODEL.BACKBONE.lower()=="resnet152":
            backbone = resnet152 
        model = ResNetWrapper(backbone, cfg.NUM_CLASSES, model_name = cfg.MODEL.BACKBONE, in_channels=in_channels, fc_in=fc_in)
    elif "densenet" in cfg.MODEL.BACKBONE.lower():
        if cfg.MODEL.BACKBONE=="densenet121":
            backbone = densenet121 
        elif cfg.MODEL.BACKBONE=="densenet161":
            backbone = densenet161 
        elif cfg.MODEL.BACKBONE=="densenet201":
            backbone = densenet201 
        model = DenseNetWrapper(backbone, cfg.NUM_CLASSES, model_name = cfg.MODEL.BACKBONE, in_channels=in_channels, fc_in=fc_in)
    
    # remove last softmax layer
    # model = nn.Sequential(*list(model.children())[:-1])
    # model = model.to(device)

    model = load_ckp(cfg.TEST.MODEL_PATH, model)
    # ckp = torch.load(cfg.TEST.MODEL_PATH)
    # # update backbone state_dict
    # backbone_state_dict = {k.replace('backbone.', ''): v for k, v in ckp.items() if k.startswith('backbone.')}
    # model.backbone.load_state_dict(backbone_state_dict)
    # # update midfeat state_dict
    # midfeat_state_dict = {k.replace('mid_feat.', ''): v for k, v in ckp.items() if k.startswith('mid_feat.')}
    # model.mid_feat.load_state_dict(midfeat_state_dict)

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
            _, X_test_kd, y_test, filename = data
            # X_test = X_test.to(device).float()
            X_test_kd = X_test_kd.to(device).float()

            # print("[!] X_test", X_test.shape)    
            outputs = model(X_test_kd)

            preds.append(outputs.data.cpu().numpy())
            onehot_labels.append(y_test.data.cpu().numpy())
            filenames.append(filename)
                
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

