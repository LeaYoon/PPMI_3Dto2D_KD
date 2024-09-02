import numpy as np

from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
from opt.Interpolation3D import ImageInterpolator
from data.DataIO import load_npz, ImageDataIO
from opt.StandardScalingData import StandardScalingData, MinMaxScalingData
from opt.preprocess import get_dynamic_image, get_entropy_based_weighted_image, get_img_entropy

class FPCIT2DDataset(data.Dataset):
    def __init__(self, masking, early_fusion, comp_method="rank", view=None, channel_padding=None, _set="val", seed=None, PD_cls=True, resize=(64, 64, 64), only_striatum=False):
        """
        :: X: 3D dataset (N, C, D, H, W) 
        :: y: label (N, )
        :: masking: if True, __getitem__ returns masked X.
        :: early_fusion
        :: comp_method: one of two option between "rank" or "entropy"
        :: view: if early_fusion is False, returned input is single 2D image, argument 'view' should indicate the axis to compress the image.
        :: channel_padding: default is None. If the value is one of either "copy" or "zero", single channel is changed into 3 dimension with zero padding or copied channels
        """
        
        assert early_fusion==True or view is not None, "If early_fusion is False, view should be specified!"
        
        self.masking = masking
        self.early_fusion = early_fusion
        self.comp_method = comp_method
        target_path = '/home/project/datasets/PPMI_SPECT_816PD212NC.npz'
        self.D_axial_X_data, self.D_coronal_X_data, self.D_saggital_X_data, self.y_data, self.data_filename, self.label_name, self.class_name = load_npz(target_path)
        
        self.channel_padding = channel_padding
        self._set = _set
        self.seed = seed
        self.view = view
        self.resize = resize
        self.only_striatum = only_striatum
        
        D, H, W = self.resize
        if D!=64 or H!=64 or W!=64:
            imgip = ImageInterpolator(is2D=False, num_channel=1, target_size=self.resize)
            self.D_axial_X_data = np.array([imgip.interpolateImage(_X_data).reshape(D, H, W, 1) for _X_data in self.D_axial_X_data])
            self.D_coronal_X_data = np.array([imgip.interpolateImage(_X_data).reshape(D, H, W, 1) for _X_data in self.D_coronal_X_data])
            self.D_saggital_X_data = np.array([imgip.interpolateImage(_X_data).reshape(D, H, W, 1) for _X_data in self.D_saggital_X_data])
            
        # Label check
        # if PD_cls:   
        #     tmp_y_data = self.y_data.copy()   
        #     self.D_axial_X_data, self.y_data, _, _ = filter_X_with_label_index(self.D_axial_X_data, tmp_y_data, [0, 1])
        #     self.D_coronal_X_data, self.y_data, _, _ = filter_X_with_label_index(self.D_coronal_X_data, tmp_y_data, [0, 1])
        #     self.D_saggital_X_data, self.y_data, _, _ = filter_X_with_label_index(self.D_saggital_X_data, tmp_y_data, [0, 1])
        #     self.data_filename, self.y_data, _, _ = filter_X_with_label_index(self.data_filename, tmp_y_data, [0, 1]) 
        
        # if self.masking:
        #     self.D_axial_X_data = self.D_axial_X_data*self.get_mask("axial")
        #     self.D_coronal_X_data = self.D_coronal_X_data*self.get_mask("coronal")
        #     self.D_saggital_X_data = self.D_saggital_X_data*self.get_mask("saggital")


        # 2D Projection
        if self.early_fusion:
            channel_padding=None
            
            _axial = self.project_2D(self.D_axial_X_data)
            _coronal = self.project_2D(self.D_coronal_X_data)
            _saggital = self.project_2D(self.D_saggital_X_data)
            self.X = np.concatenate([_axial, _coronal, _saggital], axis=3) # (B, 64, 64, 3)
        else:
            if self.view=="axial":
                # print("!!!!!!!!!!!! self.D_axial_X_data", self.D_axial_X_data.shape)
                self.X = self.D_axial_X_data
            elif self.view == "coronal":
                self.X = self.D_coronal_X_data 
            elif self.view == "sagittal":
                self.X = self.D_saggital_X_data
            self.X = self.project_2D(self.X)
        
        if self.channel_padding is not None:
            X_shape = self.X.shape
            
            if X_shape[3] == 1:
                pad = np.zeros([X_shape[0], X_shape[1], X_shape[2], 2])

                if self.channel_padding=="copy":
                    _input = [self.X.copy(), self.X.copy(), self.X.copy()]
                elif self.channel_padding=="zero":
                    _input = [self.X.copy(), pad]
                
                # print("[!] self.X.shape", self.X.shape)
                # print("[!] pad.shape", pad.shape)
                self.X = np.concatenate(_input, axis=3)
                

        # data split
        self.get_set()

        if self.masking:
            if self.early_fusion:
                
                self.axial_mask = self.project_2D(np.expand_dims(self.get_mask("axial"), 0))
                self.axial_mask[self.axial_mask>0]=1

                self.coronal_mask = self.project_2D(np.expand_dims(self.get_mask("coronal"), 0))
                self.coronal_mask[self.coronal_mask>0]=1

                self.saggital_mask = self.project_2D(np.expand_dims(self.get_mask("saggital"), 0))
                self.saggital_mask[self.saggital_mask>0]=1

                self.concat_mask = np.concatenate([self.axial_mask, self.coronal_mask, self.saggital_mask], axis=3)

                self.X = self.X*self.concat_mask

                if self.only_striatum:
                    h_range, w_range = self.get_masked_img(self.X)
                    self.X = self.X[:, h_range[0]:h_range[1]+1, w_range[0]:w_range[1]+1, :] # (1, 16, 26, 1)
                    imgip = ImageInterpolator(is2D=True, num_channel=3, target_size=self.resize)
                    self.X = np.array([imgip.interpolateImage(img) for img in self.X])
            
            else:
                if self.view=="axial":
                    # print("!!!!!!!!!!!! self.mask", self.get_mask("axial").shape)
                    self.axial_mask = self.project_2D(np.expand_dims(self.get_mask("axial"), 0))
                    self.axial_mask[self.axial_mask>0]=1

                    self.X = self.X*self.axial_mask
                elif self.view=="coronal":
                    self.coronal_mask = self.project_2D(np.expand_dims(self.get_mask("coronal"), 0))
                    self.coronal_mask[self.coronal_mask>0]=1

                    self.X = self.X*self.coronal_mask
                elif self.view=="sagittal":
                    self.saggital_mask = self.project_2D(np.expand_dims(self.get_mask("saggital"), 0))
                    self.saggital_mask[self.saggital_mask>0]=1

                    self.X = self.X*self.saggital_mask

                if self.only_striatum:
                    h_range, w_range = self.get_masked_img(self.X)
                    self.X = self.X[:, h_range[0]:h_range[1]+1, w_range[0]:w_range[1]+1, :]
                    imgip = ImageInterpolator(is2D=True, num_channel=1, target_size=self.resize)
                    self.X = np.array([imgip.interpolateImage(img) for img in self.X])
        return

    def get_set(self):
        # data split
        D_X_train, D_X_test, D_y_train, D_y_test, D_X_train_pid, D_X_test_pid = train_test_split(self.X,
                                                                                                  self.y_data,
                                                                                                  self.data_filename,
                                                                                                  test_size=0.2,
                                                                                                  stratify=self.y_data,
                                                                                                  shuffle=True,
                                                                                                  random_state=self.seed)


        D_X_train_sub, D_X_val, D_y_train_sub, D_y_val, D_X_train_pid_sub, D_X_val_pid = train_test_split(D_X_train,
                                                                                                  D_y_train,
                                                                                                  D_X_train_pid,
                                                                                                  test_size=0.25,
                                                                                                  stratify=D_y_train,
                                                                                                  shuffle=True,
                                                                                                  random_state=self.seed)

        # D_X_train_norm = D_X_train_sub/255.0
        # D_X_val_norm = D_X_val/255.0
        # D_X_test_norm = D_X_test/255.0
        s, scaled_D_X_train = StandardScalingData(D_X_train_sub, keep_dim=True, train=True, scaler=None)
        _, scaled_D_X_val = StandardScalingData(D_X_val, keep_dim=True, train=True, scaler=s)
        _, scaled_D_X_test = StandardScalingData(D_X_test, keep_dim=True, train=True, scaler=s)

        # D_X_train_norm = scaled_D_X_train[:,:,:,None]
        # D_X_val_norm = scaled_D_X_val[:,:,:,None]
        # D_X_test_norm = scaled_D_X_test[:,:,:,None]
        D_X_train_norm = scaled_D_X_train
        D_X_val_norm = scaled_D_X_val
        D_X_test_norm = scaled_D_X_test

        # D_y_train_sub = to_categorical(D_y_train_sub, 2)
        # D_y_val = to_categorical(D_y_val, 2)
        # D_y_test = to_categorical(D_y_test, 2)
        # _D_y_train_sub = D_y_train_sub[:,None]
        # _D_y_val = D_y_val[:,None]
        # _D_y_test = D_y_test[:,None]
        print("D_X_train_norm", D_X_train_norm.shape)
        print("D_X_val_norm", D_X_val_norm.shape)
        print("D_X_test_norm", D_X_test_norm.shape)

        # print("_D_y_train_sub", _D_y_train_sub.shape)
        # print("_D_y_val", _D_y_val.shape)
        # print("_D_y_test", _D_y_test.shape)
        if self._set == "train":
            self.X = D_X_train_norm
            self.y_data = D_y_train_sub
            self.data_filename = D_X_train_pid_sub
        elif self._set == "val":
            self.X = D_X_val_norm
            self.y_data = D_y_val
            self.data_filename = D_X_val_pid
        elif self._set == "test":
            self.X = D_X_test_norm
            self.y_data = D_y_test
            self.data_filename = D_X_test_pid
        else:
          print("_set argument should be among 'train', 'val', or 'test'")
          exit(1)
        return

    def __len__(self):
        return len(self.X)
    
    def get_mask(self, view):
        # input masking
        # mask_path = r'/content/drive/MyDrive/python_project/[핵의학과]_3DImage_Classification/FP-CIT data/AAL3v1_79_95_68.nii'
        mask_path = r'/home/project/datasets/AAL3v1.nii'

        extention = "nii"
        dataDim = "3D" # 3D
        modeling = "3D"
        D, H, W = self.resize
        idio = ImageDataIO(extention, dataDim, instanceIsOneFile=True, modeling=modeling, view=view)
        st_mask = idio.read_file(mask_path)
        st_mask = idio.convert_PIL_to_numpy([st_mask])[0]

        postprocessed_st_mask = (st_mask/255)*170
        postprocessed_st_mask[postprocessed_st_mask==1]=0
        postprocessed_st_mask[postprocessed_st_mask==75]=1
        postprocessed_st_mask[postprocessed_st_mask==76]=1
        postprocessed_st_mask[postprocessed_st_mask==77]=1
        postprocessed_st_mask[postprocessed_st_mask==78]=1
        postprocessed_st_mask[postprocessed_st_mask!=1]=0

        imgip = ImageInterpolator(is2D=False, num_channel=1, target_size=self.resize)
        resized_postprocessed_st_mask = imgip.interpolateImage(postprocessed_st_mask) 
        return resized_postprocessed_st_mask.reshape(D, H, W, 1)

    def project_2D(self, X):
        X_shape = X.shape
        if self.comp_method == "rank":
            tmp_X = []
            for img in X:
                tmp_X.append(get_dynamic_image(img, normalized=True)) # (B, 64, 64, 64, 1)
            X = np.expand_dims(np.array(tmp_X), axis=-1) # (B, 64, 64, 1)
        
        elif self.comp_method == "entropy":
            tmp_X = []
            for img in X:
                tmp_X.append(get_entropy_based_weighted_image(img[:,:,:,0])) # (B, 64, 64, 1)
            X = np.array(tmp_X)
        else:
            NotImplemented
        return X       
 
    def __getitem__(self, idx): 
        X = self.X[idx].copy()

        # if self.channel_padding:         
        #     X_shape = self.X.shape
            
        #     if X_shape[3] == 1:
        #         # print("[!] X_shape", X_shape)
        #         pad = np.zeros([X_shape[0], X_shape[1], X_shape[2], 2])
        #         _input = [X, pad]
        #         # _input = [X, X, X]
        #         print("[!] _input", np.array(_input).shape)
        #         X = np.concatenate(_input, axis=2)
                
        #     X = torch.from_numpy(X).permute(2, 0, 1)
        # else :
        #     X = torch.from_numpy(X).permute(2, 0, 1) # (C, H, W)
        X = torch.from_numpy(X).permute(2, 0, 1) # (C, H, W)
        y = torch.from_numpy(np.asarray(self.y_data[idx]))
        # print(X.size(), y.size())
        return X, y, self.data_filename[idx]
    
    def get_masked_img(self, target_imgs):
        _, H, W, _ = target_imgs.shape
        
        h_indice = []
        w_indice = []
        
        for h_ind in range(H):
            if target_imgs[:, h_ind, :, :].sum()==0:
                continue
            else:
                h_indice.append(h_ind)
                break

        for h_ind in range(H-1, -1, -1):
            if target_imgs[:, h_ind, :, :].sum()==0:
                continue
            else:
                h_indice.append(h_ind)
                break

        for w_ind in range(W):
            if target_imgs[:, :, w_ind, :].sum()==0:
                continue
            else:
                w_indice.append(w_ind)
                break

        for w_ind in range(W-1, -1, -1):
            if target_imgs[:, :, w_ind, :].sum()==0:
                continue
            else:
                w_indice.append(w_ind)
                break

        return h_indice, w_indice