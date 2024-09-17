# -*- coding: utf-8 -*-
# time: 2024/6/17 16:19
# file: norm.py
# author: Shuai


import netCDF4 as nc
import numpy as np
import os
from tqdm import tqdm
import argparse


def calculate_band_statistics(nc_file_path, args):
    # 读取nc.txt文件中的文件名
    with open(nc_file_path, 'r') as txt_file:
        nc_file_names = txt_file.read().splitlines()

    base = r'J:\research\soil_moistur_retrieval\SMCI_1km\10cm\SMCI_1km_{}_{}'.format(args.year, args.depth)

    values_pixel_sum = 0
    pixels = 0
    # 遍历每个nc文件
    for file_name in tqdm(nc_file_names):
        file = os.path.join(base, file_name) + '.nc'
        dataset = nc.Dataset(file)

        band_data = dataset.variables['SMCI'][:]
        band_data = np.array(band_data)
        band_data = np.nan_to_num(band_data, copy=False, nan=0, posinf=0, neginf=0)
        band_data[band_data == -999] = 0

        values_pixel_sum += np.sum(band_data)
        pixels += band_data.shape[0]*band_data.shape[1]*band_data.shape[2]
    mean = values_pixel_sum / pixels
    numerator = 0
    for file_name in tqdm(nc_file_names):
        file = os.path.join(base, file_name) + '.nc'
        dataset = nc.Dataset(file)

        band_data = dataset.variables['SMCI'][:]
        band_data = np.array(band_data)
        band_data = np.nan_to_num(band_data, copy=False, nan=0, posinf=0, neginf=0)
        band_data[band_data == -999] = 0

        numerator += np.sum(np.square(band_data.flatten() - mean))
    std = np.sqrt(numerator / pixels)

    np.save('{}_{}_SM_mean_and_std.npy'.format(args.year, args.depth), [mean, std])


# 用法示例
parser = argparse.ArgumentParser()
parser.add_argument("--year", type=str, default='2012')
parser.add_argument("--depth", type=str, default='10cm')
args = parser.parse_args()
nc_file_path = r'H:\soil_moistur_retrieval\to_chaosuan\{}_{}_names.txt'.format(args.year, args.depth)
calculate_band_statistics(nc_file_path, args)

