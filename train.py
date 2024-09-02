import os
import sys



import argparse
import random
import numpy as np
import pandas as pd
import yaml
from easydict import EasyDict
import torch

from configs.config import cfg
from configs.config import cfg_from_file

from classification.classify_fpcit3d import classify_fpcit3d
from classification.classify_fpcit2d import classify_fpcit2d
from classification.eval_fpcit3d import evaluate_fpcit3d
from classification.eval_fpcit2d import evaluate_fpcit2d


def setup():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for individual model training")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    
    parser.add_argument("--modality", type=str, help="2D or 3D")
    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument("--backbone", type=str, help="backbone name e.g. [PDNet] / [resnet18] / [ViT2D]")
    parser.add_argument("--batchsize", type=int, default=32, help="batch size")
    parser.add_argument("--lr_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--pretrained", type=lambda x: (str(x).lower() == 'true'), default=False, help="only for 2D resnet")
    parser.add_argument("--masking", type=lambda x: (str(x).lower() == 'true'), default=False, help="whether input is masked or not")
    parser.add_argument("--only_striatum", type=lambda x: (str(x).lower() == 'true'), default=False, help="whether input is striatum region or not")

    parser.add_argument("--comp_method", type=str, default="rank", help="either [rank] or [entropy]")
    parser.add_argument("--view", type=str, default="axial", help="among [axial], [sagittal] or [coronal]")
    parser.add_argument("--early_fusion", type=lambda x: (str(x).lower() == 'true'), default=False, help="using 3 planes")
    parser.add_argument("--joint_fusion", type=lambda x: (str(x).lower() == 'true'), default=False, help="using 3 planes")
    parser.add_argument("--rep_exec", type=lambda x: (str(x).lower() == 'true'), default=False, help="whether repeated execution is applied or not")
    parser.add_argument("--add_rep_exec", type=lambda x: (str(x).lower() == 'true'), default=False, help="whether additionally repeated execution is applied or not")
    parser.add_argument("--random_train", type=lambda x: (str(x).lower() == 'true'), default=False, help="whether fixed seed is applied or not for training")
    parser.add_argument("--num_exec", type=int, default=50, help="the number of repeated execution")
    parser.add_argument("--seed", type=int, default=0, help="selected seed for experiment")

    # evaluation
    parser.add_argument("--only_eval", type=bool, default=False, help="only evaluation is applied")
    parser.add_argument("--eval_model_name", type=str, default='', help="evaluation model name")
    parser.add_argument("--eval_dataset", type=str, default='fpcit', help="dataset only for evaluation")
    parser.add_argument("--test_batchsize", type=int, default=32, help="batch size for test")
    parser.add_argument("--rep_eval", type=bool, default=False, help="multiple model is evaluated")
    
    # ViT parameters
    parser.add_argument("--vit_patch_size", type=int, default=8, help="ViT/patch_size")
    parser.add_argument("--vit_dim", type=int, default=16, help="ViT/dim, D")
    parser.add_argument("--vit_mlp_dim", type=int, default=16, help="ViT/mlp_dim, hidden units of MLP")
    parser.add_argument("--vit_depth", type=int, default=6, help="ViT/depth, depth of encoder")
    parser.add_argument("--vit_heads", type=int, default=8, help="ViT/heads, #multi-heads")

    # KD
    parser.add_argument("--save_model", type=lambda x: (str(x).lower() == 'true'), default=False, help="whether model is saved after training")
    parser.add_argument('--teacher_model', type=str, default=None, help='teacher model for KD')
    parser.add_argument('--kd_loss', type=str, default=None, help='indicate KD Loss if needed')
    parser.add_argument("--only_kd", type=lambda x: (str(x).lower() == 'true'), default=False, help="whether kd loss is only used if kd is activated")
    parser.add_argument("--save_test_graph", type=lambda x: (str(x).lower() == 'true'), default=False, help="whether similarity graph is saved or not")
    parser.add_argument("--kd_temperature", type=int, default=1, help="KD/temperature")
    parser.add_argument("--kd_loss_ratio", type=float, default=1.0, help="kd loss ratio")
    parser.add_argument("--cls_loss_ratio", type=float, default=1.0, help="cls loss ratio")


    args = parser.parse_args()

    # replace arguments with config file 
    cfg_from_file(args.cfg)
    # replace arguments with shell condition
    cfg.EXP_NAME=args.exp_name
    cfg.MODEL.MODALITY=args.modality
    cfg.MODEL.BACKBONE=args.backbone
    cfg.MODEL.PRETRAINED_RESNET=args.pretrained
    cfg.TRAIN.BATCH_SIZE=args.batchsize
    cfg.TRAIN.LEARNING_RATE=args.lr_rate
    cfg.INPUT.MASKING = args.masking
    cfg.INPUT.ONLY_STRIATUM = args.only_striatum
    cfg.INPUT.COMP_METHOD = args.comp_method
    cfg.INPUT.VIEW = args.view
    cfg.INPUT.EARLY_FUSION = args.early_fusion
    cfg.INPUT.JOINT_FUSION = args.joint_fusion
    cfg.REP_EXEC = args.rep_exec
    cfg.ADD_REP_EXEC = args.add_rep_exec
    cfg.RANDOM_TRAIN = args.random_train
    cfg.NUM_EXEC = args.num_exec
    cfg.SEED = args.seed

    # test
    cfg.ONLY_EVAL = args.only_eval
    cfg.RESTORE_FROM = args.eval_model_name
    cfg.TEST.DATASET = args.eval_dataset
    cfg.TEST.BATCH_SIZE = args.test_batchsize
    cfg.TEST.REP_EVAL = args.rep_eval


    cfg.MODEL.VIT_PATCH_SIZE = args.vit_patch_size
    cfg.MODEL.VIT_DIM = args.vit_dim
    cfg.MODEL.VIT_MLP_DIM = args.vit_mlp_dim
    cfg.MODEL.VIT_DEPTH = args.vit_depth
    cfg.MODEL.VIT_HEADS = args.vit_heads
    cfg.MODEL.SAVE_MODEL = args.save_model

    cfg.KD.TEACHER_MODEL = args.teacher_model
    cfg.KD.LOSS = args.kd_loss
    cfg.KD.ONLY_KD = args.only_kd
    cfg.KD.TEMPERATURE = args.kd_temperature
    cfg.KD.LOSS_RATIO = args.kd_loss_ratio
    cfg.TRAIN.CLS_LOSS_WEIGHT = args.cls_loss_ratio
    cfg.KD.SAVE_TEST_GRAPH = args.save_test_graph

    # 인자가 의도대로 들어오는지 확인
    # print(f"[!] comp_entropy:{args.comp_method} | kd_temp: {args.kd_temperature} |  kd ratio: {args.kd_loss_ratio}")
    
    # 확인 후에는 프로그램 종료
    # import sys
    # sys.exit()

    return 


def main():
    print("Launch FPCIT BASE code")
    setup()

    print("cuda available: ", torch.cuda.is_available())

    print("[!] cfg")
    print(cfg)
    
    print()
    print()
    
    if not cfg.RANDOM_TRAIN:
        """
        https://hoya012.github.io/blog/reproducible_pytorch/
        """
        torch.manual_seed(cfg.SEED) # model param. is controlled
        if torch.cuda.device_count()>1:
            torch.cuda.manual_seed_all(cfg.SEED)
        else:
            torch.cuda.manual_seed(cfg.SEED) # controlled learning convergence
        np.random.seed(cfg.SEED) # controlled numpy and sklearn operation
        random.seed(cfg.SEED) # controlled torchvision transform
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False

    if cfg.ONLY_EVAL:
        print("ONLY EVALUATION")
        if cfg.RESTORE_FROM is None:
            raise ValueError("[!] cfg.EVAL_MODEL_NAME vaiable should be set with cfg.ONLY_EVAL==True")
        
        if cfg.TEST.REP_EVAL:
            results = dict()
            _exp_root = cfg.EXP_ROOT
            os.makedirs(os.path.join(_exp_root, cfg.EXP_NAME), exist_ok=True)
            
            if os.path.isdir(cfg.RESTORE_FROM) :
                child_list = os.listdir(cfg.RESTORE_FROM)
                for child in child_list:
                    if os.path.isdir(os.path.join(cfg.RESTORE_FROM, child)):
                        cfg.EXP_ROOT = os.path.join(_exp_root, cfg.EXP_NAME, child)
                        os.makedirs(cfg.EXP_ROOT, exist_ok=True)
                        cfg.TEST.MODEL_PATH = os.path.join(cfg.RESTORE_FROM, child, 'model', 'best_model.pth')

                        # get the model path as *.pth
                        if os.path.exists(cfg.TEST.MODEL_PATH) is False:
                            model_cands = os.listdir(os.path.join(cfg.RESTORE_FROM, child, 'model'))
                            def extract_idx(x):
                                return int(x.split('_')[-1].split('.')[0][1:])
                            # get the latest model
                            best_model_name = sorted(model_cands, key=extract_idx)[-1]

                            cfg.TEST.MODEL_PATH = os.path.join(cfg.RESTORE_FROM, child, 'model', best_model_name)
                            # check whether the model path ends with .pth
                            if cfg.TEST.MODEL_PATH.endswith('.pth') is False:
                                raise ValueError("[!] Model path should be ended with .pth")
                    else:
                        continue
                    if cfg.MODEL.MODALITY=="2D":
                        res = evaluate_fpcit2d(cfg)
                    elif cfg.MODEL.MODALITY=="3D":
                        res = evaluate_fpcit3d(cfg) 
                    for item in cfg.METRICS:
                        try:
                            results[item].append(res[item])
                        except KeyError:
                            results[item]=[]
                            results[item].append(res[item])
                            pass
            else:
                cfg.TEST.MODEL_PATH = cfg.RESTORE_FROM
                if cfg.MODEL.MODALITY=="2D":
                    res = evaluate_fpcit2d(cfg)
                elif cfg.MODEL.MODALITY=="3D":
                    res = evaluate_fpcit3d(cfg) 
                for item in cfg.METRICS:
                    try:
                        results[item].append(res[item])
                    except KeyError:
                        results[item]=[]
                        results[item].append(res[item])
                        pass            
                
            # save the results 
            res_savepath = os.path.join(_exp_root, cfg.EXP_NAME, "total_raw_results.xlsx")
            res_df = pd.DataFrame(results)
            res_df.to_excel(res_savepath)

            total_res_savepath = os.path.join(_exp_root, cfg.EXP_NAME, "total_results.txt")
            with open(total_res_savepath, "a") as f:
                f.write("Total mean ACC: %s\n" % np.mean(results["ACCURACY"]))
                f.write("Total mean RECALL: %s\n" % np.mean(results["RECALL"]))
                f.write("Total mean Weighted F1: %s\n" % np.mean(results["WEIGHTED_F1"]))

            print("Total Mean ACC:", np.mean(results["ACCURACY"]))
            print("Total Mean RECALL:", np.mean(results["RECALL"]))
            print("Total Mean WEIGHTED_F1:", np.mean(results["WEIGHTED_F1"]))
            print()
        
        return

    if cfg.MODEL.MODALITY=="3D":
        print("START 3D Model Experiment")

        if cfg.REP_EXEC:
            results = dict()
            _exp_root = cfg.EXP_ROOT
            os.makedirs(os.path.join(_exp_root, cfg.EXP_NAME), exist_ok=True)

            for seed in range(cfg.NUM_EXEC):

                cfg.EXP_ROOT = os.path.join(_exp_root, cfg.EXP_NAME, str(seed))
                os.makedirs(os.path.join(cfg.EXP_ROOT), exist_ok=True)

                torch.manual_seed(seed) # model param. is controlled
                torch.cuda.manual_seed(seed) # controlled learning convergence
                np.random.seed(seed) # controlled numpy and sklearn operation
                random.seed(seed) # controlled torchvision transform

                res = classify_fpcit3d(cfg)
                for item in cfg.METRICS:
                    try:
                        results[item].append(res[item])
                    except KeyError:
                        results[item]=[]
                        results[item].append(res[item])
                        pass
            
            # save the results 
            res_savepath = os.path.join(_exp_root, cfg.EXP_NAME, "total_raw_results.xlsx")
            res_df = pd.DataFrame(results)
            res_df.to_excel(res_savepath)

            total_res_savepath = os.path.join(_exp_root, cfg.EXP_NAME, "total_results.txt")
            with open(total_res_savepath, "a") as f:
                f.write("Total mean ACC: %s\n" % np.mean(results["ACCURACY"]))
                f.write("Total mean RECALL: %s\n" % np.mean(results["RECALL"]))
                f.write("Total mean Weighted F1: %s\n" % np.mean(results["WEIGHTED_F1"]))

            print("Total Mean ACC:", np.mean(results["ACCURACY"]))
            print("Total Mean RECALL:", np.mean(results["RECALL"]))
            print("Total Mean WEIGHTED_F1:", np.mean(results["WEIGHTED_F1"]))
            print()
        
        else:
            classify_fpcit3d(cfg)

    elif cfg.MODEL.MODALITY=="2D":
        print("START 2D Model Experiment")
        
        if cfg.REP_EXEC:
            results = dict()
            _exp_root = cfg.EXP_ROOT
            os.makedirs(os.path.join(_exp_root, cfg.EXP_NAME), exist_ok=True)

            if cfg.ADD_REP_EXEC:
                start_idx = cfg.NUM_EXEC
                end_idx = cfg.NUM_EXEC*2
            else:
                start_idx = 0
                end_idx = cfg.NUM_EXEC

            for seed in range(start_idx, end_idx):
            
                cfg.EXP_ROOT = os.path.join(_exp_root, cfg.EXP_NAME, str(seed))
                os.makedirs(os.path.join(cfg.EXP_ROOT), exist_ok=True)

                torch.manual_seed(seed) # model param. is controlled
                torch.cuda.manual_seed(seed) # controlled learning convergence
                np.random.seed(seed) # controlled numpy and sklearn operation
                random.seed(seed) # controlled torchvision transform

                res = classify_fpcit2d(cfg)
                for item in cfg.METRICS:
                    try:
                        results[item].append(res[item])
                    except KeyError:
                        results[item]=[]
                        results[item].append(res[item])
                        pass
                
            # save the results 
            res_savepath = os.path.join(_exp_root, cfg.EXP_NAME, "total_raw_results.xlsx")
            res_df = pd.DataFrame(results)
            res_df.to_excel(res_savepath)

            total_res_savepath = os.path.join(_exp_root, cfg.EXP_NAME, "total_results.txt")
            with open(total_res_savepath, "a") as f:
                f.write("Total mean ACC: %s\n" % np.mean(results["ACCURACY"]))
                f.write("Total mean RECALL: %s\n" % np.mean(results["RECALL"]))
                f.write("Total mean Weighted F1: %s\n" % np.mean(results["WEIGHTED_F1"]))

            print("Total Mean ACC:", np.mean(results["ACCURACY"]))
            print("Total Mean RECALL:", np.mean(results["RECALL"]))
            print("Total Mean WEIGHTED_F1:", np.mean(results["WEIGHTED_F1"]))
            print()
            
        else:
            classify_fpcit2d(cfg)

    return

if __name__ == "__main__":

    # time check
    from time import time
    start_time = time()
    main()
    end_time = time()
    print(f"Execution Time: {end_time-start_time}")
    