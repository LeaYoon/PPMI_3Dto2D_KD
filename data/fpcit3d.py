import numpy as np

from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
from opt.Interpolation3D import ImageInterpolator
from data.DataIO import load_npz, ImageDataIO
from opt.StandardScalingData import StandardScalingData, MinMaxScalingData

class FPCIT3DDataset(data.Dataset):
    def __init__(self, masking, _set="val", seed=None, PD_cls=True, resize=(64, 64, 64), \
                 ch_padding=None, only_striatum=False, transform=None):
        """
        :: masking: if True, __getitem__ returns masked X. else, it returns original X.
        :: _set: _set should be adopted in ["train", "val", "test"] 
        """
        self.masking = masking

        target_path = '/home/project/datasets/PPMI_SPECT_816PD212NC.npz'
        self.D_axial_X_data, _, _, self.y_data, self.data_filename, self.label_name, self.class_name = load_npz(target_path)
        self._set = _set
        self.seed = seed
        self.resize = resize
        self.ch_padding = ch_padding
        self.only_striatum = only_striatum
        self.transform = transform
        D, H, W = self.resize
        if D!=64 or H!=64 or W!=64:
            imgip = ImageInterpolator(is2D=False, num_channel=1, target_size=self.resize)
            self.D_axial_X_data = np.array([imgip.interpolateImage(_X_data).reshape(D, H, W, 1) for _X_data in self.D_axial_X_data])
            # self.D_coronal_X_data = np.array([imgip.interpolateImage(_X_data).reshape(D, H, W, 1) for _X_data in self.D_coronal_X_data])
            # self.D_saggital_X_data = np.array([imgip.interpolateImage(_X_data).reshape(D, H, W, 1) for _X_data in self.D_saggital_X_data])
          
        # Label check
        # if PD_cls:   
        #     tmp_y_data = self.y_data.copy()   
        #     self.D_axial_X_data, self.y_data, _, _ = filter_X_with_label_index(self.D_axial_X_data, tmp_y_data, [0, 1])
        #     self.data_filename, self.y_data, _, _ = filter_X_with_label_index(self.data_filename, tmp_y_data, [0, 1])
        
        if self.masking:
            self.D_axial_X_data = self.D_axial_X_data*self.get_mask("axial")

            if self.only_striatum:
                d_range, h_range, w_range = self.get_masked_vol(self.get_mask("axial"))
                self.D_axial_X_data = self.D_axial_X_data[:, d_range[0]:d_range[1]+1, h_range[0]:h_range[1]+1, w_range[0]:w_range[1]+1, :]
            
        self.X = self.D_axial_X_data

        # data split
        self.get_set()
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
            # return resized_postprocessed_st_mask.reshape(64, 64, 64, 1)
            return resized_postprocessed_st_mask.reshape(D, H, W, 1)

    def __getitem__(self, idx): 
        X = self.X[idx]

        X_shape = X.shape
        if self.ch_padding == "zero" and X_shape[3] == 1:
            pad = np.zeros([X_shape[0], X_shape[1], X_shape[2], 2])
            X = np.concatenate([X, pad], axis=3)
            X = torch.from_numpy(X).permute(3, 0, 1, 2)
            # X = torch.from_numpy(self.X[idx]).permute(3, 0, 1, 2) # (C, H, W)
        elif self.ch_padding == None:
            X = torch.from_numpy(X).permute(3, 0, 1, 2)
        else: 
            print(self.ch_padding, "is not supported")
            raise NotImplemented
        
        y = torch.from_numpy(np.asarray(self.y_data[idx]))
        # print(X.size(), y.size())
        if self.transform is not None:
            X = self.transform(X)
        return X, y, self.data_filename[idx]

    def get_masked_vol(self, target_vol):
        D, H, W, _ = target_vol.shape
        d_indice = []
        h_indice = []
        w_indice = []
        
        for d_ind in range(D):
            if target_vol[d_ind, :, :, 0].sum()==0:
                continue
            else:
                d_indice.append(d_ind)
                break

        for d_ind in range(D-1, -1, -1):
            if target_vol[d_ind, :, :, 0].sum()==0:
                continue
            else:
                d_indice.append(d_ind)
                break
            
        for h_ind in range(H):
            if target_vol[:, h_ind, :, 0].sum()==0:
                continue
            else:
                h_indice.append(h_ind)
                break

        for h_ind in range(H-1, -1, -1):
            if target_vol[:, h_ind, :, 0].sum()==0:
                continue
            else:
                h_indice.append(h_ind)
                break

        for w_ind in range(W):
            if target_vol[:, :, w_ind, 0].sum()==0:
                continue
            else:
                w_indice.append(w_ind)
                break

        for w_ind in range(W-1, -1, -1):
            if target_vol[:, :, w_ind, 0].sum()==0:
                continue
            else:
                w_indice.append(w_ind)
                break

        return d_indice, h_indice, w_indice