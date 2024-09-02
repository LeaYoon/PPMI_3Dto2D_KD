import os
import numpy as np
from sklearn.metrics import recall_score, classification_report, confusion_matrix

import torch
from torch import nn
from torchsummary import summary
import torchio as tio

from data.fpcit3d import FPCIT3DDataset

from data.DataIO import load_npz
from easydict import EasyDict
from vit_pytorch.vit_3d import ViT

from opt.Interpolation3D import ImageInterpolator
from models.CNN3D import PDNet
from models.ViT import ViTWrapper
from models.ResNet3D import ResNetWrapper, resnet18, resnet34, resnet50, resnet101, resnet152
from models.DenseNet3D import DenseNetWrapper, densenet121, densenet161, densenet201

from opt.preprocess import to_categorical
from opt.lr_scheduler import CosineAnnealingWarmUpRestarts
from evaluation import get_performances, plot_trend, save_pred_label

def classify_fpcit3d(cfg):
    
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
    # view = cfg.INPUT.VIEW #"coronal" "axial" "saggital"
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

    train_dataset = FPCIT3DDataset(masking=masking, _set="train", seed=11, resize = cfg.INPUT.RESIZE, ch_padding=channel_padding, only_striatum=only_striatum, transform=transform)
    val_dataset = FPCIT3DDataset(masking=masking, _set="val", seed=11, resize = cfg.INPUT.RESIZE, ch_padding=channel_padding, only_striatum=only_striatum)
    test_dataset = FPCIT3DDataset(masking=masking, _set="test", seed=11, resize = cfg.INPUT.RESIZE, ch_padding=channel_padding, only_striatum=only_striatum)

    batch_size = cfg.TRAIN.BATCH_SIZE
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)
    
    if channel_padding:
        in_channels = 3
    else:
        in_channels = 1

    # Model Load
    if cfg.MODEL.BACKBONE=="PDNet":
        v = PDNet(in_channels)
    elif cfg.MODEL.BACKBONE=="ViT3D":
        v = ViT(image_size=cfg.INPUT.RESIZE[1],
        frames = cfg.INPUT.RESIZE[0],
        image_patch_size = cfg.MODEL.VIT_PATCH_SIZE, # 8*8 grid
        frame_patch_size= cfg.MODEL.VIT_PATCH_SIZE, # 8
        num_classes=2,
        dim=cfg.MODEL.VIT_DIM, # 128
        depth=cfg.MODEL.VIT_DEPTH, # 6
        heads=cfg.MODEL.VIT_HEADS, # 16
        mlp_dim=cfg.MODEL.VIT_MLP_DIM, # 32
        channels=in_channels) # 128
        
        v = ViTWrapper(v)
    elif "resnet" in cfg.MODEL.BACKBONE.lower():
        if cfg.MODEL.BACKBONE=="resnet18":
            backbone = resnet18 
            fc_in = 512
        elif cfg.MODEL.BACKBONE=="resnet34":
            backbone = resnet34 
        elif cfg.MODEL.BACKBONE=="resnet50":
            backbone = resnet50 
        elif cfg.MODEL.BACKBONE=="resnet101":
            backbone = resnet101 
        elif cfg.MODEL.BACKBONE=="resnet152":
            backbone = resnet152 
        v = ResNetWrapper(backbone, cfg.NUM_CLASSES, model_name = cfg.MODEL.BACKBONE, in_channels=in_channels, fc_in=fc_in)
    elif "densenet" in cfg.MODEL.BACKBONE.lower():
        if cfg.MODEL.BACKBONE=="densenet121":
            backbone = densenet121 
        elif cfg.MODEL.BACKBONE=="densenet161":
            backbone = densenet161 
        elif cfg.MODEL.BACKBONE=="densenet201":
            backbone = densenet201 
        v = DenseNetWrapper(backbone, cfg.NUM_CLASSES, model_name = cfg.MODEL.BACKBONE, in_channels=in_channels, fc_in=fc_in)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device", device)
    v = v.to(device)

    criterion = nn.CrossEntropyLoss()
    lr_rate = cfg.TRAIN.LEARNING_RATE
    # optimizer = torch.optim.Adam(v.parameters(), lr=lr_rate) # 0.00005 # 0.001
    optimizer = torch.optim.RAdam(v.parameters(), lr=0) # 0.00005 # 0.001
    scheduler = CosineAnnealingWarmUpRestarts(optimizer=optimizer, T_0=20, T_mult=1, T_up=10, eta_max=0.01, gamma=0.5)
    # summary(v, (3, cfg.INPUT.RESIZE[0], cfg.INPUT.RESIZE[1],cfg.INPUT.RESIZE[2])) #  >_<!

    ### Training Model ###
    
    ### Experimental param

    num_epochs = cfg.TRAIN.NUM_EPOCH #500 #
    print_step = cfg.TRAIN.PRINT_STEP # 10 # ?
    patience_limit = cfg.TRAIN.PATIENCE_LIMIT # 20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = v.to(device)

    best_loss=cfg.TRAIN.BEST_LOSS_INIT
    train_history = []
    val_history = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        train_loss = 0.0
        val_loss = 0.0 
        for i, data in enumerate(train_loader):
            
            # get the inputs; data is a list of [inputs, labels, filenames]
            X_train, y_train, filenames = data
            X_train = X_train.to(device).float()
            # zero the parameter gradients
            optimizer.zero_grad()
            # print("X_train.shape", X_train.shape )
            # forward + backward + optimize
            v = v.cuda()
            outputs = v(X_train) # (64, 1, 64, 64)
            
            # print("outputs", outputs)
            # print()
            loss = criterion(outputs, y_train.cuda().long())
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ')
                exit(1)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print statistics
            train_loss += loss.item()
        train_loss =train_loss/X_train.size(0)
        train_history.append(train_loss)    

        # again no gradients needed
        with torch.no_grad():
            preds = []  
            for data in val_loader:
                X_val, y_val, filename = data
                X_val = X_val.to(device).float()
                
                outputs = v(X_val)
                _, predictions = torch.max(outputs.data, 1) # returns (max_values, indice)
                loss = criterion(outputs, y_val.cuda())
                val_loss+=loss.item()
                preds.append(predictions.data.cpu().numpy())
        val_loss = val_loss/X_val.size(0)
        val_history.append(val_loss)    
        
        if epoch % print_step == 0:
            print(f'[Epoch: {epoch + 1}] train loss: {train_history[-1]:.3f}, val loss: {val_history[-1]:.3f}')
        
        # early stopping check
        if val_loss > best_loss: # loss is not improved
            patience_check += 1

            if patience_check >= patience_limit: 
                print("[!] Training is terminated by early stopping !!!")
                break

        else: # loss가 개선된 경우
            best_loss = val_loss
            patience_check = 0

    print('Finished Training')

    history = EasyDict()
    history.history=EasyDict()
    history.history["loss"] = train_history
    history.history["val_loss"] = val_history

    # cfg.PROJECT_ROOT = project_root
    # cfg.EXP_ROOT = cfg.PROJECT_ROOT / 'experiments'
    # cfg.EXP_NAME = ''
    # cfg.DATA_DIRECTORY_SOURCE = cfg.PROJECT_ROOT / 'dataset'

    # #'d:\\윤혜민\\[핵의학과]\\FPCCIT_Ex_3D_Image_Classification'
    # main_dir = cfg.PROJECT_ROOT
    # proj_dirname = cfg.proj_dirname
    # ex_dirname = cfg.ex_dirname #"Ex_2D_cls_ViT_EarlyFusion_epoch500_v1.0"

    # if not os.path.isdir(cfg.EXP_ROOT) and not cfg.REP_EXEC:
    #     os.mkdir(cfg.EXP_ROOT)
    # if not os.path.isdir(os.path.join(cfg.EXP_ROOT, cfg.EXP_NAME)) and not cfg.REP_EXEC:
    #     os.mkdir(os.path.join(cfg.EXP_ROOT, cfg.EXP_NAME))
    os.makedirs(cfg.EXP_ROOT, exist_ok=True)
    if not cfg.REP_EXEC:
        os.makedirs(os.path.join(cfg.EXP_ROOT, cfg.EXP_NAME), exist_ok=True)
        D_EXPERIMENTAL_DIR = os.path.join(cfg.EXP_ROOT, cfg.EXP_NAME)
    else:
        D_EXPERIMENTAL_DIR = os.path.join(cfg.EXP_ROOT)
    
    ### Model Save ###
    if cfg.MODEL.SAVE_MODEL :
        checkpoint_path = os.path.join(D_EXPERIMENTAL_DIR, "model")
        os.makedirs(checkpoint_path, exist_ok=True)

        save_dict = {
            'model': v.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(save_dict, os.path.join(checkpoint_path, f"{cfg.EXP_NAME}_E{epoch}.pth"))

    plot_trend(history, save_path=os.path.join(D_EXPERIMENTAL_DIR, "learning_curve.png"))

    # again no gradients needed
    with torch.no_grad():
        preds = []  
        onehot_labels = []
        filenames = []
        for data in test_loader:
            X_test, y_test, filename = data
            X_test = X_test.to(device).float()
            
            outputs = v(X_test)
            preds.append(outputs.data.cpu().numpy())
            onehot_labels.append(y_test.data.cpu().numpy())
            filenames.append(filename)
                
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

