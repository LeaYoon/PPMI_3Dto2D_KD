import os
import pathlib
import numpy as np
import yaml
from easydict import EasyDict


project_root=pathlib.Path(__file__).resolve().parents[1]
print("[!] project_root / in config.py", project_root) # /home/project

cfg = EasyDict()

cfg.PROJECT_ROOT = project_root
cfg.EXP_ROOT = cfg.PROJECT_ROOT / 'experiments'
cfg.EXP_NAME = ''

cfg.DATA_DIRECTORY_SOURCE = cfg.PROJECT_ROOT / 'dataset'
# cfg.DATA_CSV_MATCH = cfg.DATA_DIRECTORY_SOURCE / 'matched_csv_woDMN.yml'
# cfg.DATA_CHECK_INTEGRITY = True

cfg.NUM_CLASSES = 2
cfg.SEED=0
cfg.ONLY_EVAL=False
cfg.EVAL_MODEL_NAME = ''
cfg.REP_EXEC=False
cfg.ADD_REP_EXEC=False
cfg.RANDOM_TRAIN=False
cfg.NUM_EXEC = 3
cfg.METRICS = ['ACCURACY', 'RECALL', 'WEIGHTED_F1']

cfg.INPUT = EasyDict()
cfg.INPUT.EARLY_FUSION=False
cfg.INPUT.JOINT_FUSION = False
cfg.INPUT.COMP_METHOD="rank" # "entropy" or "rank"
cfg.INPUT.VIEW = "axial"
cfg.INPUT.CHANNEL_SIZE = 1
cfg.INPUT.CHANNEL_PADDING=None
cfg.INPUT.MASKING = True
cfg.INPUT.ONLY_STRIATUM = False
cfg.INPUT.RESIZE = (64, 64, 64)

cfg.TRAIN = EasyDict()
# cfg.TRAIN.INPUT_SHAPE = (128, 128, 1)
cfg.TRAIN.BATCH_SIZE = 512 # 32
cfg.TRAIN.NUM_EPOCH = 500
cfg.TRAIN.CHANNEL_SIZE = 1
cfg.TRAIN.LEARNING_RATE = 0.001
cfg.TRAIN.BEST_LOSS_INIT = 10**4
cfg.TRAIN.PRINT_STEP = 10
cfg.TRAIN.PATIENCE_LIMIT = 50 #10 # 50
# cfg.TRAIN.RESIZE = (109, 109, 109)
cfg.TRAIN.EARLY_STOP = 1000
cfg.TRAIN.SAVE_PRED_EVERY = 100
cfg.TRAIN.CLS_LOSS_WEIGHT = 20.0

cfg.MODEL = EasyDict()
cfg.MODEL.MODALITY="3D"
cfg.MODEL.BACKBONE = "ViT2D"

cfg.MODEL.PRETRAINED_RESNET = False


# ViT
cfg.MODEL.VIT_PATCH_SIZE = 8
cfg.MODEL.VIT_DIM = 16
cfg.MODEL.VIT_MLP_DIM = 16
cfg.MODEL.VIT_DEPTH = 6
cfg.MODEL.VIT_HEADS = 8

cfg.KD = EasyDict()
cfg.KD.TEACHER_MODEL = ""
cfg.KD.LOSS = "CE"
cfg.KD.ONLY_KD = False
cfg.KD.TEMPERATURE = 1
cfg.KD.LOSS_RATIO = 1.0
cfg.KD.SAVE_TEST_GRAPH = False

# TEST CONFIGS
cfg.TEST = EasyDict()
cfg.TEST.BATCH_SIZE = 64
cfg.TEST.RESTORE_FROM = ''
cfg.TEST.DATASET = 'fpcit' # or 'av133'
cfg.TEST.REP_EVAL=False

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v

def yaml_load(file_path):
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=yaml.Loader)

def cfg_from_file(filename):
    yaml_cfg = EasyDict(yaml_load(filename))
    _merge_a_into_b(yaml_cfg, cfg)
