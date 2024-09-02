import numpy as np

from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
from opt.Interpolation3D import ImageInterpolator
from data.DataIO import load_npz, ImageDataIO
from opt.StandardScalingData import StandardScalingData, MinMaxScalingData
from opt.preprocess import get_dynamic_image, get_entropy_based_weighted_image, get_img_entropy

class AV133IntegratedDataset(data.Dataset):
    def __init__(self, early_fusion=False, comp_method="rank", view=None, seed=None, PD_cls=True, \
                 resize=(64, 64, 64), channel_padding=None, only_striatum=False, midslice_gap=1):
        """
        this dataset class is only for AV133 as test dataset.
        
        :: masking: if True, __getitem__ returns masked X. else, it returns original X.
        
        """
        # self.masking = masking
        self.early_fusion = early_fusion
        self.view = view  
        self.comp_method = comp_method

        target_path = '/home/project/datasets/PPMI_F18AV133_36PD1NC.npz'
        
        self.D_axial_X_data, self.D_coronal_X_data, self.D_saggital_X_data, self.y_data, self.data_filename, self.label_name, self.class_name = load_npz(target_path)
        
        # self._set = _set
        # self.seed = seed
        self.resize = resize
        self.channel_padding = channel_padding # channel_padding
        self.only_striatum = only_striatum
        self.midslice_gap  = midslice_gap
        D, H, W = self.resize
        if D!=64 or H!=64 or W!=64:
            imgip = ImageInterpolator(is2D=False, num_channel=1, target_size=self.resize)
            self.D_axial_X_data = np.array([imgip.interpolateImage(_X_data).reshape(D, H, W, 1) for _X_data in self.D_axial_X_data])
            self.D_coronal_X_data = np.array([imgip.interpolateImage(_X_data).reshape(D, H, W, 1) for _X_data in self.D_coronal_X_data])
            self.D_saggital_X_data = np.array([imgip.interpolateImage(_X_data).reshape(D, H, W, 1) for _X_data in self.D_saggital_X_data])
        
        # print("[!] self.D_axial_X_data.shape", self.D_axial_X_data.shape)
        # print("[!] self.D_coronal_X_data.shape", self.D_coronal_X_data.shape)
        # print("[!] self.D_saggital_X_data.shape", self.D_saggital_X_data.shape)


        # Label check
        # if PD_cls:   
        #     tmp_y_data = self.y_data.copy()   
        #     self.D_axial_X_data, self.y_data, _, _ = filter_X_with_label_index(self.D_axial_X_data, tmp_y_data, [0, 1])
        #     self.data_filename, self.y_data, _, _ = filter_X_with_label_index(self.data_filename, tmp_y_data, [0, 1])
        
        # 2D Projection
        if self.early_fusion:
            channel_padding=None
            
            _axial = self.project_2D(self.D_axial_X_data)
            _coronal = self.project_2D(self.D_coronal_X_data)
            _saggital = self.project_2D(self.D_saggital_X_data)
            self.Dim2_X = np.concatenate([_axial, _coronal, _saggital], axis=3) # (B, 64, 64, 3)
        else:
            if self.view=="axial":
                self.Dim2_X = self.D_axial_X_data
            elif self.view == "coronal":
                self.Dim2_X = self.D_coronal_X_data 
            elif self.view == "sagittal":
                self.Dim2_X = self.D_saggital_X_data
            
            self.Dim2_X = self.project_2D(self.Dim2_X)  # (1028, 32, 32, 32, 1)
        if self.channel_padding is not None:
            X_shape = self.Dim2_X.shape
            
            if X_shape[3] == 1:
                pad = np.zeros([X_shape[0], X_shape[1], X_shape[2], 2])

                if self.channel_padding=="copy":
                    _input = [self.X.copy(), self.X.copy(), self.X.copy()]
                elif self.channel_padding=="zero":
                    _input = [self.X.copy(), pad]
                
                # print("[!] self.X.shape", self.X.shape)
                # print("[!] pad.shape", pad.shape)
                self.Dim2_X = np.concatenate(_input, axis=3)   
        # # masking for 2D
        # if self.masking:
        #     if self.early_fusion:
                
        #         self.axial_mask = self.project_2D(np.expand_dims(self.get_mask("axial"), 0))
        #         self.axial_mask[self.axial_mask>0]=1

        #         self.coronal_mask = self.project_2D(np.expand_dims(self.get_mask("coronal"), 0))
        #         self.coronal_mask[self.coronal_mask>0]=1

        #         self.saggital_mask = self.project_2D(np.expand_dims(self.get_mask("saggital"), 0))
        #         self.saggital_mask[self.saggital_mask>0]=1

        #         self.concat_mask = np.concatenate([self.axial_mask, self.coronal_mask, self.saggital_mask], axis=3)

        #         self.Dim2_X = self.Dim2_X*self.concat_mask

        #         if self.only_striatum:
        #             h_range, w_range = self.get_masked_img(self.X)
        #             self.Dim2_X = self.Dim2_X[:, h_range[0]:h_range[1]+1, w_range[0]:w_range[1]+1, :] # (1, 16, 26, 1)
        #             imgip = ImageInterpolator(is2D=True, num_channel=3, target_size=self.resize)
        #             self.Dim2_X = np.array([imgip.interpolateImage(img) for img in self.Dim2_X])
            
        #     else:
        #         if self.view=="axial":
        #             # print("!!!!!!!!!!!! self.mask", self.get_mask("axial").shape)
        #             self.axial_mask = self.project_2D(np.expand_dims(self.get_mask("axial"), 0))
        #             self.axial_mask[self.axial_mask>0]=1

        #             self.Dim2_X = self.Dim2_X*self.axial_mask
        #         elif self.view=="coronal":
        #             self.coronal_mask = self.project_2D(np.expand_dims(self.get_mask("coronal"), 0))
        #             self.coronal_mask[self.coronal_mask>0]=1

        #             self.Dim2_X = self.Dim2_X*self.coronal_mask
        #         elif self.view=="sagittal":
        #             self.saggital_mask = self.project_2D(np.expand_dims(self.get_mask("saggital"), 0))
        #             self.saggital_mask[self.saggital_mask>0]=1

        #             self.Dim2_X = self.Dim2_X*self.saggital_mask

        #         if self.only_striatum:
        #             h_range, w_range = self.get_masked_img(self.Dim2_X)
        #             self.Dim2_X = self.X[:, h_range[0]:h_range[1]+1, w_range[0]:w_range[1]+1, :]
        #             imgip = ImageInterpolator(is2D=True, num_channel=1, target_size=self.resize)
        #             self.Dim2_X = np.array([imgip.interpolateImage(img) for img in self.Dim2_X])

        # # masking for 3D 
        # if self.masking:
        #     self.D_axial_X_data = self.D_axial_X_data*self.get_mask("axial")

        #     if self.only_striatum:
        #         d_range, h_range, w_range = self.get_masked_vol(self.get_mask("axial"))
        #         self.D_axial_X_data = self.D_axial_X_data[:, d_range[0]:d_range[1]+1, h_range[0]:h_range[1]+1, w_range[0]:w_range[1]+1, :]
            
        self.Dim3_X = self.D_axial_X_data

        if self.comp_method == "midslice":
            self.Dim2_X = np.concatenate([self.Dim2_X[:,:,:,0,:], self.Dim2_X[:,:,:,1,:], self.Dim2_X[:,:,:,2,:]])
            self.Dim3_X = np.concatenate([self.Dim3_X, self.Dim3_X, self.Dim3_X])
            self.y_data = np.concatenate([self.y_data, self.y_data, self.y_data])
            self.data_filename = np.concatenate([self.data_filename, self.data_filename, self.data_filename])
        return
    
    def __len__(self):
        return len(self.Dim2_X)

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


    def project_2D(self, X):
        X_shape = X.shape
        # print("[!] project_2D", X.shape) # project_2D (1028, 32, 32, 32, 1)
        print("[!] project_2D", self.comp_method)
        if self.comp_method == "rank":
            tmp_X = []
            for img in X:
                
                # res = get_dynamic_image(img, normalized=True)
                # print("rank", res.shape) # (32, 32)
                tmp_X.append(get_dynamic_image(img, normalized=True)) # (B, 64, 64, 64, 1)
            X = np.expand_dims(np.array(tmp_X), axis=-1) # (B, 64, 64, 1)
            print("[!] project_2D, X.shape", X.shape)
        elif self.comp_method == "entropy":
            tmp_X = []
            for img in X:
                tmp_X.append(get_entropy_based_weighted_image(img[:,:,:,0])) # (B, 64, 64, 1)
            X = np.array(tmp_X)
        elif self.comp_method == "2D+e":
            
            d_range, _, _ = self.get_masked_vol(self.get_mask(self.view))
            d_mid = int((d_range[0]+d_range[1])/2)

            tmp_X = []
            for img in X:
                _slices = img[d_mid-1:d_mid+2,:,:,0].transpose(1,2,0)
            
                tmp_X.append(_slices)
            X = np.array(tmp_X) # (B, 64, 64, 1)
            
        elif self.comp_method == "midslice":
            
            d_range, _, _ = self.get_masked_vol(self.get_mask(self.view))
            d_mid = int((d_range[0]+d_range[1])/2)
            # h_mid = int((h_range[0]+h_range[1])/2)
            # w_mid = int((w_range[0]+w_range[1])/2)

            tmp_X = []
            for img in X:
                # res = img[d_mid,:,:,0]
                # print("[!] midslice", res.shape) # midslice (32, 32)
                _slices = img[d_mid-self.midslice_gap:d_mid+self.midslice_gap+1,:,:,0].transpose(1,2,0)
                tmp_X.append(_slices)
            X = np.expand_dims(np.array(tmp_X), axis=-1) # (B, 64, 64, 1)
            
        else:
            NotImplemented
        return X      

    def __getitem__(self, idx): 
        Dim2_X = self.Dim2_X[idx].copy()
        # get 2D sample

        Dim2_X = torch.from_numpy(Dim2_X).permute(2, 0, 1) # (C, H, W)
    
        # get 3D sample
        Dim3_X = self.Dim3_X[idx].copy()

        Dim3_X_shape = Dim3_X.shape
        if self.channel_padding == "zero" and Dim3_X_shape[3] == 1:
            pad = np.zeros([Dim3_X_shape[0], Dim3_X_shape[1], Dim3_X_shape[2], 2])
            Dim3_X = np.concatenate([Dim3_X, pad], axis=3)
            Dim3_X = torch.from_numpy(Dim3_X).permute(3, 0, 1, 2)
            
        elif self.channel_padding == None:
            Dim3_X = torch.from_numpy(Dim3_X).permute(3, 0, 1, 2)
        else: 
            print(self.channel_padding, "is not supported")
            raise NotImplemented
        
        y = torch.from_numpy(np.asarray(self.y_data[idx]))
        # print(X.size(), y.size())
        return Dim2_X, Dim3_X, y, self.data_filename[idx]

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