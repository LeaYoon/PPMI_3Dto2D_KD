# PPMI_3Dto2D_KD
This is official github for our Paper "Enhancement and Evaluation for deep learning-based classification of volumetric neuroimaging with 3D-to-2D Knowledge Distillation"

# Intallation
This code is tested under torch==2.0.1 and Geforce RTX 3080

# Dataset Preparation
Datasets is precomputed as *.npy as follows:

[PPMI_SPECT_816PD212NC.npy]()

[PPMI_F18AV133_36PD1NC.npy]()

Dataset path is initially hard-coded in dataloader.
If you hope to run this code without any modification, the datasets path should be set as follow.

### A typical top-level directory layout

    .
    ├── datasets                # dataset root directory
            ├── PPMI_SPECT_816PD212NC.npy     # 123I-DaTscan SPECT dataset
            ├── PPMI_F18AV133_36PD1NC.npy     # 18F-AV133 PET dataset
    ├── configs
    ├── data
    ├── classification
    ├── models
    ├── opt
    └── README.md

### Training


```
# Training 3D teacher network
python train.py --cfg ./configs/fpcit_base.yml --rep_exec True --num_exec 20 --modality 3D --exp_name ResNet18_3D_teacher --save_model True --backbone resnet18

# Training 2D student with 3D-to-2D KD
python train.py --cfg ./configs/fpcit_base.yml \
            --batchsize 64 --rep_exec True --num_exec 20 --modality 2D --save_model True \
            --exp_name axial_2D+e_S_DenseNet121_T_DenseNet121_KDTEST_rep_exec_GS_lf_metric_t5_bs64_loss1:${_ratio}_240901 \
            --backbone DenseNet121 --view axial --comp_method 2D+e --early_fusion False \
            --joint_fusion True --masking False --teacher_model /home/project/experiments/DenseNet121_3D_teacher_240829_model/8/model/DenseNet121_3D_teacher_240829_model_E60.pth \
            --kd_loss GS_lf --kd_temperature 5 --cls_loss_ratio 1.0 --kd_loss_ratio 5.0
```

### Test

```
# TEST model on 18F-AV133 PET dataset
python train.py --cfg ./configs/fpcit_base.yml --only_eval True --rep_eval True --backbone resnet18 --eval_dataset av133 --modality 3D --exp_name eval_av133_ResNet_3D_teacher_240830 --eval_model_name /home/project/experiments/ResNet18_3D_teacher_240901_model
```

