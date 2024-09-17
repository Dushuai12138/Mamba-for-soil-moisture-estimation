# -*- coding: utf-8 -*-
# time: 2024/6/4 16:58
# file: stack_bands.py
# author: Shuai


import os
from osgeo import gdal
import numpy as np
from tqdm import tqdm
import argparse


def stack_tiff_files(input_folder, output_folder, args):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有子文件夹
    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]
    process = open('stack_process.txt', 'w')

    process.close()
    # 获取第一个子文件夹的所有文件名（假设所有子文件夹中的文件名相同）
    # example_subfolder = subfolders[0]
    # tiff_files = [f for f in os.listdir(example_subfolder) if f.endswith('.tif')]
    with open('{}_{}_names.txt'.format(args.year, args.depth), "r") as f:
        total_names = f.readlines()
    tiff_files = [i[:-1]+'.tif' for i in total_names]
    num = 1
    for tiff_name in tqdm(tiff_files):

        # 初始化一个列表来存储所有子文件夹中的同名TIFF文件
        tiff_paths = []

        for subfolder in subfolders:
            tiff_path = os.path.join(subfolder, tiff_name)
            if os.path.exists(tiff_path):
                tiff_paths.append(tiff_path)

        # 读取并堆叠TIFF文件
        src_datasets = []
        for fp in tiff_paths:
            src_datasets.append(gdal.Open(fp))

        bands_data = []
        for ds in src_datasets:
            for i in range(ds.RasterCount):
                bd = ds.GetRasterBand(i + 1).ReadAsArray()
                bands_data.append(bd)

        bands_names = []
        for idx, ds in enumerate(src_datasets):
            name = tiff_paths[idx].split('/')[-2]
            for i in range(ds.RasterCount):
                bd = ds.GetRasterBand(i+1).GetDescription()
                bands_names.append('{}-{}'.format(name, bd))

        # 创建输出文件
        driver = gdal.GetDriverByName('GTiff')
        out_path = os.path.join(output_folder, tiff_name)
        # print(src_datasets)
        out_dataset = driver.Create(out_path, src_datasets[0].RasterXSize, src_datasets[0].RasterYSize, len(bands_names),
                                    gdal.GDT_Float32)

        # 写入每个波段
        for idx, band_data in enumerate(bands_data, start=1):
            out_dataset.GetRasterBand(idx).WriteArray(np.array(band_data))
            out_dataset.GetRasterBand(idx).SetDescription(bands_names[idx-1])

        # 复制元数据
        out_dataset.SetGeoTransform(src_datasets[0].GetGeoTransform())
        out_dataset.SetProjection(src_datasets[0].GetProjection())

        # 关闭所有文件
        for ds in src_datasets:
            ds = None
        out_dataset = None

        process = open('stack_process.txt', 'a')
        process.write('{} images have been staked\n'.format(num))
        process.close()
        num = num + 1


parser = argparse.ArgumentParser()
parser.add_argument("--year", type=str, default='2012')
parser.add_argument("--depth", type=str, default='10cm')
args = parser.parse_args()
# 使用该函数
input_folder = 'tiles/2012/10cm'  # 替换为实际输入文件夹路径
output_folder = 'tiles/2012_stack/10cm'  # 替换为实际输出文件夹路径
stack_tiff_files(input_folder, output_folder, args)



