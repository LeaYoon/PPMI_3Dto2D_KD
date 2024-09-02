import os
import sys
import math 

import numpy as np
import pydicom
from nibabel import load as nib_load # NiFti
from nibabel import ecat
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from opt.custom_exception import CustomException

class ImageDataIO():
    # arrange dimension for each type of view (axial / saggital / coronal)
    def __init__(self, extention, dataDim, instanceIsOneFile, modeling="2D", view=None):
        """
        :param extention: extention of file to read
        :param dataDim: data dim to treat, '2D' of image or '3D' of image, in a file;
        :param instanceIsOneFile: how is an instance stored for one instance; True'one_file' / 'multiple_files'
        :param modeling: how to handle an instance, e.g 2D or 3D
        :param view:
        """

        # if is2D == True and view is not None:
        #     print("is2D variable can't have view property!")
        #     return
        self._extention = extention
        self._dataDimInOneFile = dataDim
        self._instanceIsOneFile = instanceIsOneFile
        self._modeling = modeling
        self._view = view
        return

    # handling one file
    def read_file(self, source_path):
        if self._extention == "jpg" or self._extention == "jpeg" or self._extention == "png":
            return self._read_popular_img(source_path)
        elif self._extention == "dcm" or self._extention == "dicom":
            return self._read_dicom_img(source_path)
        elif self._extention == "nii" or self._extention == "nifti":
            return self._read_nifti_img(source_path)
        elif self._extention == "v" or self._extention == "ecat":
            return self._read_ecat_img(source_path)

    # reading jpeg or png
    def _read_popular_img(self, source_path):
        """
        if one file's type to read is either jpg or png, we don't have to consider the case the file have 3D data.
        :param source_path: input path for one instance to read from jpg or png file
        :return: a image or images consisting of one instance
        """
        if self._dataDimInOneFile == "2D" and self._instanceIsOneFile:
            img = Image.open(source_path)
            return img

        elif self._dataDimInOneFile == "2D" and not self._instanceIsOneFile:
            child_paths = [os.path.join(source_path, path) for path in os.listdir(source_path)]
            imgs = []
            for child_path in child_paths:
                img = Image.open(child_path)
                imgs.append(img)

            return imgs
        elif self._dataDimInOneFile == "3D":
            raise CustomException("when input file's extention is jpg/png, the case the input is 3D is not considered")

    # reading dicom
    def _read_dicom_img(self, source_path):
        """
        :param source_path: input path for one instance to read from dicom file
        :return: a image or images consisting of one instance
        """
        if self._dataDimInOneFile == "2D" and self._instanceIsOneFile:
            dcm_img = np.array(pydicom.read_file(source_path).pixel_array)
            return Image.fromarray(dcm_img)
        elif self._dataDimInOneFile == "2D" and not self._instanceIsOneFile:  # source_path is list of dcm files for one subject
            child_paths = [os.path.join(source_path, path) for path in os.listdir(source_path)]
            dcm_imgs = []
            for child_path in child_paths:
                try:
                    dcm_img = pydicom.read_file(child_path)
                except InvalidDicomError:
                    dcm_img = pydicom.read_file(child_path, force=True)
                dcm_imgs.append(dcm_img)
            # sorting and save
            dcm_imgs.sort(key=lambda x: int(x.ImagePositionPatient[2]))
            dcm_imgs = np.array([dcm_img.pixel_array for dcm_img in dcm_imgs])
            if self._view == "axial" or self._view == "transaxial":
                dcm_imgs = dcm_imgs
            elif self._view == "coronal":
                dcm_imgs = np.rot90(dcm_imgs, k=1, axes=(1, 2))  #
                dcm_imgs = np.rot90(dcm_imgs, k=3, axes=(0, 1))
            elif self._view == "saggital":
                dcm_imgs = np.rot90(dcm_imgs, k=1, axes=(0, 2))  #

            # intensity normalization
            _min = dcm_imgs.min()
            _max = dcm_imgs.max()
            dcm_imgs = (dcm_imgs-_min)/_max
            dcm_imgs = dcm_imgs*255

            return [Image.fromarray(dcm_img_pixels) for dcm_img_pixels in dcm_imgs]
        elif self._dataDimInOneFile == "3D" and self._instanceIsOneFile:
            dcm_img = np.array(pydicom.read_file(source_path).pixel_array)
            return Image.fromarray(dcm_img)
        else:
            raise CustomException(
                "the state that _dataDimInOneFile is 3D and _instanceIsOneFile is False is not defined")

    # reading nifti
    def _read_nifti_img(self, source_path):
        """
        :param source_path: input path for one instance to read from nitfi file
        :return: a image or images consisting of one instance
        """
        """
        :param source_path: 
        :return: list of Image obj 
        """
        nib_img = nib_load(source_path)
        nib_img = np.array(nib_img.get_data())
        if self._dataDimInOneFile == "2D" and self._instanceIsOneFile:
            nib_img = np.rot90(nib_img, k=1, axes=(0, 1))
            return Image.fromarray(nib_img)
        elif self._dataDimInOneFile == "3D" and not self._instanceIsOneFile:
            child_paths = [os.path.join(source_path, path) for path in os.listdir(source_path)]
            nii_imgs = []
            for child_path in child_paths:
                nii_img = nib_load(child_path)
                nii_imgs.append(nii_img)
            nii_imgs = np.array([nii_img.get_data() for nii_img in nii_imgs])
            if self._view == "axial" or self._view == "transaxial":
                nii_imgs = np.rot90(nii_imgs, k=1, axes=(0, 1))  # (95, 79, 68)
            elif self._view == "sagittal":
                nii_imgs = np.rot90(nii_imgs, k=1, axes=(0, 2))  #
            elif self._view == "coronal":
                nii_imgs = np.rot90(nii_imgs, k=1, axes=(1, 2))  #
                nii_imgs = np.rot90(nii_imgs, k=3, axes=(0, 1))

            nii_imgs = nii_imgs.astype(np.uint8)
            nii_imgs = np.transpose(nii_imgs, [2, 0, 1])
            return [Image.fromarray(nii_img) for nii_img in nii_imgs]
        elif self._dataDimInOneFile == "3D" and self._instanceIsOneFile:
            if len(nib_img.shape) ==4 and nib_img.shape[3]==1:
              nib_img = nib_img[:, :, :, 0]
            if self._view == "axial" or self._view == "transaxial":
                nib_img = np.rot90(nib_img, k=1, axes=(0, 1))  # (95, 79, 68)
            elif self._view == "sagittal":
                nib_img = np.rot90(nib_img, k=1, axes=(0, 2))
            elif self._view == "coronal":
                nib_img = np.rot90(nib_img, k=1, axes=(1, 2))
                nib_img = np.rot90(nib_img, k=3, axes=(0, 1))

            # nib_img = self._change_image_intensity_range(nib_img, encoding_to_change=8)
            # nib_img = nib_img.astype(np.uint8)
            _min = nib_img.min()
            _max = nib_img.max()
            # nib_img = (nib_img-_min)/_max
            nib_img = (nib_img-_min)/(_max-_min) 
            nib_img = nib_img*255

            nib_img = np.transpose(nib_img, [2, 0, 1])

            return [Image.fromarray(img) for img in nib_img]

        else:
            raise CustomException(
                "the state that _instanceIsOneFile is False is not defined yet in Nifti type")

    # reading nifti
    def _read_ecat_img(self, source_path, rep_slice=0):
        """
        :param source_path: input path for one instance to read from nitfi file
        :return: a image or images consisting of one instance
        """
        """
        :param source_path: 
        :param rep_slice: representative slice to get.
        e.g. if rep_slice = 0, it will return first slice of 3D image (H, W, D, 0)
        :return: list of Image obj 
        """

        ecat_img = ecat.load(source_path)
        ecat_img = np.array(ecat_img.get_fdata())
        if self._dataDimInOneFile == "2D" and self._instanceIsOneFile:
            ecat_img = np.rot90(ecat_img, k=1, axes=(0, 1))
            return Image.fromarray(ecat_img)
        # elif dataDimInOneFile == "3D" and not instanceIsOneFile:
        #     child_paths = [os.path.join(source_path, path) for path in os.listdir(source_path)]
        #     ecat_imgs = []
        #     for child_path in child_paths:
        #         ecat_img = ecat.load(child_path)
        #         ecat_imgs.append(ecat_img)
        #     ecat_imgs = np.array([ecat_img.get_data() for ecat_img in ecat_imgs])

        #     if view == "axial" or view == "transaxial":
        #         nii_imgs = np.rot90(nii_imgs, k=1, axes=(0, 1))  # (95, 79, 68)
        #     elif view == "sagittal":
        #         nii_imgs = np.rot90(nii_imgs, k=1, axes=(0, 2))  #
        #     elif view == "coronal":
        #         nii_imgs = np.rot90(nii_imgs, k=1, axes=(1, 2))  #
        #         nii_imgs = np.rot90(nii_imgs, k=3, axes=(0, 1))

        #     nii_imgs = nii_imgs.astype(np.uint8)
        #     nii_imgs = np.transpose(nii_imgs, [2, 0, 1])
        #     return [Image.fromarray(nii_img) for nii_img in nii_imgs]

        elif self._dataDimInOneFile == "3D" and self._instanceIsOneFile:
            # print("[!] _read_ecat_img start", ecat_img.shape)
            if rep_slice is None and len(ecat_img.shape) ==4 and (ecat_img.shape[3]==1):
                ecat_img = ecat_img[:, :, :, 0]
            elif rep_slice is not None and len(ecat_img.shape) ==4 and (ecat_img.shape[3]>1):
                ecat_img = ecat_img[:, :, :, rep_slice]

            if self._view == "axial" or self._view == "transaxial":
                ecat_img = np.rot90(ecat_img, k=1, axes=(0, 1))  # (95, 79, 68)
            elif self._view == "sagittal":
                ecat_img = np.rot90(ecat_img, k=1, axes=(0, 2))
            elif self._view == "coronal":
                ecat_img = np.rot90(ecat_img, k=1, axes=(1, 2))
                ecat_img = np.rot90(ecat_img, k=3, axes=(0, 1))
            # print("[!] _read_ecat_img, after rot", ecat_img.shape)
            # nib_img = self._change_image_intensity_range(nib_img, encoding_to_change=8)
            # nib_img = nib_img.astype(np.uint8)
            _min = ecat_img.min()
            _max = ecat_img.max()
            ecat_img = (ecat_img-_min)/(_max-_min)
            ecat_img = ecat_img*255
            ecat_img = np.transpose(ecat_img, [2, 0, 1])

            return [Image.fromarray(img) for img in ecat_img]

        else:
            raise CustomException(
                "the state that _instanceIsOneFile is False is not defined yet in Nifti type")


    def _change_image_intensity_range(self, instance_img, encoding_to_change=8):
        instance_img = np.array(instance_img)
        _min = instance_img.min().astype(np.int64)
        _max = instance_img.max().astype(np.int64)
        _resolution_range = _max - _min

        _encoding = None

        if _resolution_range > 255:
            _encoding = 16
        else:
            _encoding = 8

        instance_img = (instance_img - _min) / 2 ** _encoding
        if encoding_to_change == 8:
            return (instance_img * (2 ** encoding_to_change)).astype(np.uint8)
        elif encoding_to_change == 16:
            return (instance_img * (2 ** encoding_to_change)).astype(np.uint16)

    def _resizing_channel(self, img_obj, resize_size, channel_size=None):
        """
        :param img_obj: img_obj have to be a list of images whether self._modeling is "2D" or "3D"
        :param resize_size:
        :param channel_size:
        :return:
        """

        if sys.getsizeof(img_obj) == 0:
            print("size of img_data object is 0")
            return None
        # if isinstance(img_obj, list):
        #     img_shape = img_obj[0].size
        # else :
        #     img_shape = img_obj.size

        # resize
        if resize_size is not None:
            if self._dataDimInOneFile == "3D":  # (N, D, H, W, C)
                sub_list = []
                for sub in img_obj:
                    imgs = [img.resize(resize_size) for img in sub]
                    sub_list.append(imgs)
                img_obj = sub_list

            elif self._dataDimInOneFile == "2D":  # (N, H, W, C)
                # img_data = np.array([cv2.resize(np.array(x), resize_size) for x in img_data]).astype(np.float32)
                if isinstance(img_obj, list):
                    img_obj = [img.resize(resize_size) for img in img_obj]
                else:
                    img_obj = img_obj.resize(resize_size)

        # check channel
        if self._dataDimInOneFile == "3D":
            sub_list = []
            for sub in img_obj:
                if channel_size is not None and channel_size == 3:
                    imgs = [img.convert("RGB") for img in sub]
                elif channel_size is not None and channel_size == 4:
                    imgs = [img.convert("RGBA") for img in sub]
                else: # if channel_size is not None and channel_size == 1: # 2
                    imgs = [img.convert("L") for img in sub]
                sub_list.append(imgs)
            img_obj = sub_list
        elif self._dataDimInOneFile == "2D":  # isinstance(img_obj, list):
            if channel_size is not None and channel_size == 1:
                img_obj = [img.convert("L") for img in img_obj]
            elif channel_size is not None and channel_size == 3:
                img_obj = [img.convert("RGB") for img in img_obj]
            elif channel_size is not None and channel_size == 4:
                img_obj = [img.convert("RGBA") for img in img_obj]
        else:  # img_obj is just one file
            if channel_size is not None and channel_size == 1:
                img_obj = img_obj.convert("L")
            elif channel_size is not None and channel_size == 3:
                img_obj = img_obj.convert("RGB")
            elif channel_size is not None and channel_size == 4:
                img_obj = img_obj.convert("RGBA")

        return img_obj

    def read_files_from_dir(self, source_dir):
        """
        :param source_dir:  input path for multiple instance to read from nitfi file
        :return:
        """
        child_full_paths = [os.path.join(source_dir, path) for path in os.listdir(source_dir)]
        file_list = []
        filename_list = []
        for child_full_path in child_full_paths:
            file_list.append(self.read_file(child_full_path))
            filename_list.append(os.path.basename(child_full_path))
        return file_list, filename_list  # list of PIL Image obj

    def convert_PIL_to_numpy(self, img_obj):

        if isinstance(img_obj, list):
            if self._dataDimInOneFile == "3D":
                sub_list = []
                for sub in img_obj:
                    img_list = []
                    for img in sub:
                        # print(np.asarray(img))
                        img_list.append(np.asarray(img))
                    img_list = np.array(img_list)
                    sub_list.append(img_list)
                return np.array(sub_list)

            elif self._dataDimInOneFile == "2D":
                return np.array([np.array(img) for img in img_obj])
        else:  # img_obj is just one instance not list
            return np.asarray(img_obj)

    def convert_numpy_to_PIL(self, img_obj, single_obj=False):
        if single_obj:
            return Image.fromarray(img_obj)
        # print("convert_numpy_to_PIL",img_obj)
        if self._dataDimInOneFile == "3D":
            sub_list = []
            for sub in img_obj:
                img_list = []
                for img in sub:
                    # print(np.asarray(img))
                    img_list.append(Image.fromarray(img))
                sub_list.append(img_list)
            return sub_list

        elif self._dataDimInOneFile == "2D":
            return [Image.fromarray(img) for img in img_obj]

    # def convertUint8(self, img_obj):
    #     """
    #     1. dicom :16bit -> the very info?? 0~65535 -> 16? 8?
    #     [0,1] / [-1,1]

    #     2. convertion into uint8


    #     :param img_obj:
    #     :param _:
    #     :return:
    #     """
    #     sub_list = []
    #     for sub in img_obj:
    #         img_list = []
    #         for img in sub:
    #             pix = np.asarray(img)
    #             pix_norm = (pix - pix.min()) / (pix.max() - pix.min())
    #             # pix_norm = (pix - pix.min()) / (65535 - pix.min())
    #             pix_norm = pix_norm * 255
    #             img_list.append(Image.fromarray(pix_norm))
    #         sub_list.append(img_list)

    #     return sub_list
    
    def _cropping(self, pil_img_list, bbox):

        if self._dataDimInOneFile == "2D" and self._instanceIsOneFile:
            return [pil_img.crop(bbox) for pil_img in pil_img_list]
        elif (self._dataDimInOneFile == "2D" and self._instanceIsOneFile is False) or \
                (self._dataDimInOneFile == "3D" and self._instanceIsOneFile):  # 4d
            sub_list = []
            for sub in pil_img_list:
                img_list = []
                for img in sub:
                    # print("img", img)
                    # print(np.asarray(img))
                    img_list.append(img.crop(bbox))

                sub_list.append(img_list)
            return sub_list
            
    def show_one_img(self, imgs, is2D=True, cmap=None):
        if is2D:
            plt.imshow(imgs, cmap)
            plt.show()
        else:
            num_imgs = len(imgs)
            board_size = math.ceil(math.sqrt(num_imgs))
            r = c = board_size
            img_ind = 0
            for r_ind in range(r):
                for c_ind in range(c):
                    if img_ind >= len(imgs):
                        plt.show()
                        return
                    c_ind_ = c_ind + 1
                    plt.subplot(r, c, r_ind * c + c_ind_)
                    plt.imshow(imgs[img_ind], cmap=cmap)
                    img_ind = img_ind + 1
            plt.show()
        return

    def show_one_img_v2(self, imgs, is2D=True, cmap=None, save_path=None):
        if is2D:
            plt.imshow(imgs, cmap)
            plt.show()
        else:
            num_imgs = len(imgs)
            board_size = math.ceil(math.sqrt(num_imgs))
            r = c = board_size
            img_ind = 0
            for r_ind in range(r):
                for c_ind in range(c):
                    if img_ind >= len(imgs):
                        plt.show()
                        return
                    # c_ind_ = c_ind+1
                    plt.subplot(r, c, r_ind * c + c_ind + 1)

                    plt.imshow(imgs[img_ind], cmap=cmap)
                    img_ind = img_ind + 1
            plt.show()
        return

    @classmethod
    def show_one_img_v3(self, img, is2D=True, cmap=None, save_path = None):
        """

        :param imgs:
        :param is2D: (X, Y) or (D, X, Y) ; not tested when img have C yet
        :param cmap:
        :return:
        """
        if is2D:  # one 2D image
            plt.imshow(img, cmap)
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            plt.close()
            plt.cla()
            plt.clf()
        else:  # one 3D image
            len_depth = len(img)
            num_rows = img.shape[1]
            num_cols = img.shape[2]
            grid_size = math.ceil(math.sqrt(len_depth))
            r_board_size = grid_size * img.shape[1]
            c_board_size = grid_size * img.shape[2]
            grid_board = np.zeros([r_board_size, c_board_size])
            img_ind = 0
            for r_grid_ind in range(grid_size):
                for c_grid_ind in range(grid_size):
                    grid_board_c_start_ind = c_grid_ind * num_cols
                    grid_board_r_start_ind = r_grid_ind * num_rows
                    grid_board[grid_board_r_start_ind:grid_board_r_start_ind + num_rows,
                    grid_board_c_start_ind:grid_board_c_start_ind + num_cols] = img[img_ind, :, :]

                    img_ind = img_ind + 1
                    if img_ind >= len(img):
                        plt.imshow(grid_board, cmap=cmap)
                        plt.colorbar()
                        if save_path :
                            plt.savefig(save_path)
                        else:
                            plt.show()
                        plt.close()
                        plt.cla()
                        plt.clf()
                        return grid_board

        return grid_board


    @classmethod
    def show_one_img_v4(self, img, is2D=True, cmap=None, save_path = None, **kwargs):
        """

        :param imgs:
        :param is2D: (X, Y) or (D, X, Y) ; not tested when img have C yet
        :param cmap:
        :return:
        """
        if is2D:  # one 2D image
            plt.imshow(img, cmap)
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            plt.close()
            plt.cla()
            plt.clf()
        else:  # one 3D image
            len_depth = len(img)
            num_rows = img.shape[1]
            num_cols = img.shape[2]
            grid_size = math.ceil(math.sqrt(len_depth))
            r_board_size = grid_size * img.shape[1]
            c_board_size = grid_size * img.shape[2]
            grid_board = np.zeros([r_board_size, c_board_size])
            img_ind = 0
            for r_grid_ind in range(grid_size):
                for c_grid_ind in range(grid_size):
                    grid_board_c_start_ind = c_grid_ind * num_cols
                    grid_board_r_start_ind = r_grid_ind * num_rows
                    grid_board[grid_board_r_start_ind:grid_board_r_start_ind + num_rows,
                    grid_board_c_start_ind:grid_board_c_start_ind + num_cols] = img[img_ind, :, :]

                    img_ind = img_ind + 1
                    if img_ind >= len(img):
                        plt.imshow(grid_board, cmap=cmap, clim=kwargs['clim'])
                        plt.colorbar()
                        if save_path :
                            plt.savefig(save_path)
                        else:
                            plt.show()
                        plt.close()
                        plt.cla()
                        plt.clf()
                        return grid_board

        return grid_board


    @classmethod
    def get_one_img(self, img, is2D=True, cmap=None, save_path = None, **kwargs):
        """

        :param imgs:
        :param is2D: (X, Y) or (D, X, Y) ; not tested when img have C yet
        :param cmap:
        :return:
        """
        if is2D:  # one 2D image
            plt.imshow(img, cmap)
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            plt.close()
            plt.cla()
            plt.clf()
        else:  # one 3D image
            len_depth = len(img)
            num_rows = img.shape[1]
            num_cols = img.shape[2]
            grid_size = math.ceil(math.sqrt(len_depth))
            r_board_size = grid_size * img.shape[1]
            c_board_size = grid_size * img.shape[2]
            grid_board = np.zeros([r_board_size, c_board_size])
            img_ind = 0
            for r_grid_ind in range(grid_size):
                for c_grid_ind in range(grid_size):
                    grid_board_c_start_ind = c_grid_ind * num_cols
                    grid_board_r_start_ind = r_grid_ind * num_rows
                    grid_board[grid_board_r_start_ind:grid_board_r_start_ind + num_rows,
                    grid_board_c_start_ind:grid_board_c_start_ind + num_cols] = img[img_ind, :, :]

                    img_ind = img_ind + 1
                    if img_ind >= len(img):
                        return grid_board

        return grid_board

# file IO modules

import numpy as np

def save_npz(save_path, *args):
    with open(save_path, 'wb') as f:
        np.savez(f, *args)          
    return

# save_npz(target_path, D_axial_X_data, D_coronal_X_data, D_saggital_X_data, D_axial_y_data, D_axial_data_filename, D_axial_label_name, D_axial_class_name)
def load_npz(load_path):
    npzfile = np.load(load_path, allow_pickle=True)
    return [npzfile[arr_name] for arr_name in npzfile.files]

