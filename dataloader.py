# -*- coding: utf-8 -*-
# time: 2024/5/31 15:25
# file: dataloader.py
# author: Shuai


from torch.utils.data import Dataset
import os
import numpy as np
from osgeo import gdal
import netCDF4 as nc


class MyCustomDataset(Dataset):
    def __init__(self, data_names, bands, dataset_path, SM_path, mean, std, mean_and_std_for_SM):
        # 加载你的数据，例如从文件中读取图像和标签
        # 在这里，你可以根据自己的需求进行数据加载和预处理
        self.data_names = data_names
        self.bands = bands
        self.dataset_path = dataset_path
        self.SM_path = SM_path
        self.mean = mean
        self.std = std
        self.mean_and_std_for_SM = mean_and_std_for_SM

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        data_name = self.data_names[idx]

        datasets = gdal.Open(os.path.join(self.dataset_path, data_name + ".tif")).ReadAsArray()
        datasets = np.nan_to_num(datasets, copy=False, nan=0, posinf=0, neginf=0)
        datasets[datasets == -999] = 0
        datasets = inputs_reshape(datasets, normalize=True, mean=self.mean, std=self.std)

        soil_moisture = nc.Dataset(os.path.join(self.SM_path, data_name + ".nc"), mode='r').variables['SMCI'][:].data
        soil_moisture = np.nan_to_num(soil_moisture, copy=False, nan=0, posinf=0, neginf=0)
        soil_moisture[soil_moisture == -999] = 0
        # soil_moisture = expand_first_dim(soil_moisture)
        soil_moisture_nor = standardize_SM(soil_moisture, self.mean_and_std_for_SM)

        sample = {
            'image': datasets,
            'label': soil_moisture_nor,
            'image_name': data_name
        }
        return sample


def inputs_reshape(data, normalize=True, mean=None, std=None):
    """
    :param data: [5507, 64, 64]
    :return: [366,32,64,64]
    2020:[1464:1481], 2012:[2562:2579]
    """
    soil_dem_lc = data[2562:2579]
    weather = np.concatenate((data[0:2562], data[2579:]), axis=0)
    weather_reshape = np.array([[weather[band*366+day] for band in range(15)] for day in range(366)])
    data = np.array([np.vstack((weather_reshape[day], soil_dem_lc)) for day in range(366)])
    if normalize:
        data = (data - mean[np.newaxis, :, np.newaxis, np.newaxis])/std[np.newaxis, :, np.newaxis, np.newaxis]

    return data


def standardize_SM(arr, mean_and_std_for_SM):
    """
    标准化数组为均值为0、方差为1的形式（Z-score标准化）。

    参数：
    arr (numpy.ndarray): 输入的数组。

    返回：
    numpy.ndarray: 标准化后的数组。
    """
    standardized_arr = (arr - mean_and_std_for_SM[0]) / mean_and_std_for_SM[1]
    return standardized_arr


def expand_first_dim(arr, target_dim=384):
    # 获取当前数组的形状
    original_shape = arr.shape

    # 检查数组是否为三维数组
    if len(original_shape) != 3:
        raise ValueError("Input array must be a 3D array")

    # 计算需要填充的数量
    pad_size = target_dim - original_shape[0]

    if pad_size > 0:
        # 构建填充参数，第一个维度填充pad_size个0，第二和第三个维度填充0
        pad_width = [(0, pad_size), (0, 0), (0, 0)]

        # 使用np.pad进行填充
        expanded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
    else:
        # 如果目标维度小于等于当前维度，直接返回原数组
        expanded_arr = arr

    return expanded_arr


class ISMN_Dataset(Dataset):
    def __init__(self, all_ISMN_list):
        # 加载你的数据，例如从文件中读取图像和标签
        self.all_ISMN_list = all_ISMN_list

    def __len__(self):
        return len(self.all_ISMN_list['ISMN'])

    def __getitem__(self, idx):
        ISMN = np.array(self.all_ISMN_list['ISMN'][idx])
        ISMN_index = np.array(self.all_ISMN_list['ISMN_index'][idx])
        variables = np.array(self.all_ISMN_list['variables'][idx])

        sample = {
            'ISMN': ISMN,
            'ISMN_index': ISMN_index,
            'variables': variables
        }
        return sample
