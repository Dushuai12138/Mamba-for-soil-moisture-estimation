# -*- coding: utf-8 -*-
# time: 2024/6/4 15:13
# file: predict.py
# author: Shuai


from load_model import load_model
import torch
from dataloader import inputs_reshape
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import netCDF4 as nc
import argparse
import os
import random
from ismn.interface import ISMN_Interface
from datetime import datetime
from utils import ismn_pre_loss, smci_pre_loss, calculate_loss
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def SM_comparasion(SMys, SMxs, precipitation, year, save_name, image_name, ismn_pre_list, smci_pre_list):
    fig = plt.subplots(figsize=(15, 4))
    # 创建 GridSpec 对象，分成2行2列
    gs = gridspec.GridSpec(2, 3, width_ratios=[3, 1, 1])  # 调整宽度比例

    ax1 = plt.subplot(gs[:, 0])
    ax1.set_ylabel("Soil Moisture(m³/m³)")
    ax1.set_xlabel("Day of Year {}".format(year))
    # 绘制第一个折线图
    mins = []
    maxs = []
    colors = ['#a73b49', "#808AB1", '#a0a89b', '#d08a7d']
    sm_order = 0
    for sm_num in SMys:
        if 'ISMN' in sm_num:
            if len(SMxs[sm_num]) == 0:
                plt.close()
                return None
            ax1.scatter(SMxs[sm_num], SMys[sm_num], label=sm_num, color=colors[sm_order], s=3)
            # 过滤掉 NaN 和 inf 值
            filtered_array = SMys[sm_num][np.isfinite(SMys[sm_num])]
            # 找到非 NaN 和非 inf 的最大值
            if filtered_array.size > 0:
                mins.append(min(filtered_array))
                maxs.append(max(filtered_array))
            sm_order = sm_order + 1
        else:
            ax1.plot(SMxs[sm_num], SMys[sm_num], label=sm_num, color=colors[sm_order], linestyle="-")
            mins.append(min(SMys[sm_num]))
            maxs.append(max(SMys[sm_num]))
            sm_order = sm_order + 1

    min_lim, max_lim = min(mins), max(maxs)
    if max_lim < 0.5:
        max_lim = 0.5
    ax1.set_ylim(0, max_lim+0.05)
    # ax1.set_ylim(0, 1)
    ax1.legend(loc='upper left', ncol=2)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Precipitation(mm)')
    ax2.set_ylim(0, 120)
    ax2.bar(np.arange(1, 367), precipitation, label='Precipitation', color='#408390', linestyle="-")
    # 添加标题和标签
    plt.title(image_name)
    # 添加图例
    ax2.legend(loc='upper right')
    # 显示图形
    plt.grid(True)

    # 小图
    # ubrmse, bias, mae, r, r2, , pre, smci
    for i in range(0, 2):
        ismn_pre1 = plt.subplot(gs[i, 1])
        data = ismn_pre_list[list(ismn_pre_list.keys())[i]]
        ismn_pre1.scatter(data[6], data[7], c='#a73b49', s=10, alpha=0.5)
        ubrmse, bias, mae, r, r2, mse = data[0], data[1], data[2], data[3], data[4], data[5]
        ismn_pre1.text(0.05, 0.90, f'ubRMSE:{ubrmse:.3f}', transform=ismn_pre1.transAxes, fontsize=10)
        ismn_pre1.text(0.05, 0.75, f'Bias:{bias:.3f}', transform=ismn_pre1.transAxes, fontsize=10)
        ismn_pre1.text(0.05, 0.60, f'MAE:{mae:.3f}', transform=ismn_pre1.transAxes, fontsize=10)
        ismn_pre1.text(0.05, 0.45, f'R:{r:.3f}', transform=ismn_pre1.transAxes, fontsize=10)
        ismn_pre1.text(0.05, 0.30, f'R²:{r2:.3f}', transform=ismn_pre1.transAxes, fontsize=10)
        ismn_pre1.set_ylim(0, max_lim)
        ismn_pre1.set_xlim(0, max_lim)
        ismn_pre1.set_ylabel('ISMN SM(m³/m³)')
        ismn_pre1.set_xlabel('Estimated SM(m³/m³)')
        ismn_pre1.plot([0, 1], [0, 1], c='#d08a7d', lw=1)
        ismn_pre1.set_title('{}'.format(list(ismn_pre_list.keys())[i]))

    for i in range(0, 2):
        smci_pre1 = plt.subplot(gs[i, 2])
        data = smci_pre_list[list(smci_pre_list.keys())[i]]
        smci_pre1.scatter(data[6], data[7], c='#a73b49', s=10, alpha=0.5)
        ubrmse, bias, mae, r, r2, mse = data[0], data[1], data[2], data[3], data[4], data[5]
        smci_pre1.text(0.05, 0.90, f'ubRMSE:{ubrmse:.3f}', transform=smci_pre1.transAxes, fontsize=10)
        smci_pre1.text(0.05, 0.75, f'Bias:{bias:.3f}', transform=smci_pre1.transAxes, fontsize=10)
        smci_pre1.text(0.05, 0.60, f'MAE:{mae:.3f}', transform=smci_pre1.transAxes, fontsize=10)
        smci_pre1.text(0.05, 0.45, f'R:{r:.3f}', transform=smci_pre1.transAxes, fontsize=10)
        smci_pre1.text(0.05, 0.30, f'R²:{r2:.3f}', transform=smci_pre1.transAxes, fontsize=10)
        smci_pre1.set_ylabel('SMCI1.0 SM(m³/m³)')
        smci_pre1.set_xlabel('Estimated SM(m³/m³)')
        smci_pre1.set_ylim(0, max_lim)
        smci_pre1.set_xlim(0, max_lim)
        smci_pre1.plot([0, 1], [0, 1], c='#d08a7d', lw=1)
        smci_pre1.set_title('{}'.format(list(smci_pre_list.keys())[i]))

    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()


def SM_scatter_comparasion(SMs, save_name):
    fig = plt.subplots(figsize=(8,8))
    # 创建 GridSpec 对象，分成2行2列
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])  # 调整宽度比例

    ismn_pre_list = dict(list(SMs.items())[:3])
    smci_pre_list = dict(list(SMs.items())[3:6])

    # 小图
    # ubrmse, bias, mae, r, r2, , pre, smci
    for i in range(1, 3):
        ismn_pre1 = plt.subplot(gs[i-1, 0])
        data = ismn_pre_list[list(ismn_pre_list.keys())[i]]
        ismn_pre1.scatter(data, ismn_pre_list['ISMN'],  c='#a73b49', s=15, alpha=0.5)
        ubrmse, bias, mae, r, r2, mse = calculate_loss(np.array(data), np.array(ismn_pre_list['ISMN']))
        ismn_pre1.text(0.05, 0.90, f'ubRMSE:{ubrmse:.3f}', transform=ismn_pre1.transAxes, fontsize=10)
        ismn_pre1.text(0.05, 0.85, f'Bias:{bias:.3f}', transform=ismn_pre1.transAxes, fontsize=10)
        ismn_pre1.text(0.05, 0.8, f'MAE:{mae:.3f}', transform=ismn_pre1.transAxes, fontsize=10)
        ismn_pre1.text(0.05, 0.75, f'R:{r:.3f}', transform=ismn_pre1.transAxes, fontsize=10)
        ismn_pre1.text(0.05, 0.70, f'R²:{r2:.3f}', transform=ismn_pre1.transAxes, fontsize=10)
        ismn_pre1.set_ylabel('ISMN SM(m³/m³)')
        ismn_pre1.set_xlabel('Estimated SM(m³/m³)')
        ismn_pre1.plot([0, 1], [0, 1], c='#d08a7d', lw=1)
        ismn_pre1.set_title('{}'.format(list(ismn_pre_list.keys())[i]))

    for i in range(1, 3):
        smci_pre1 = plt.subplot(gs[i-1, 1])
        data = smci_pre_list[list(smci_pre_list.keys())[i]]
        smci_pre1.scatter(data, smci_pre_list['SMCI1.0'], c='#a73b49', s=15, alpha=0.5)
        ubrmse, bias, mae, r, r2, mse = calculate_loss(np.array(data), np.array(smci_pre_list['SMCI1.0']))
        smci_pre1.text(0.05, 0.90, f'ubRMSE:{ubrmse:.3f}', transform=smci_pre1.transAxes, fontsize=10)
        smci_pre1.text(0.05, 0.85, f'Bias:{bias:.3f}', transform=smci_pre1.transAxes, fontsize=10)
        smci_pre1.text(0.05, 0.80, f'MAE:{mae:.3f}', transform=smci_pre1.transAxes, fontsize=10)
        smci_pre1.text(0.05, 0.75, f'R:{r:.3f}', transform=smci_pre1.transAxes, fontsize=10)
        smci_pre1.text(0.05, 0.70, f'R²:{r2:.3f}', transform=smci_pre1.transAxes, fontsize=10)
        smci_pre1.set_ylabel('SMCI1.0 SM(m³/m³)')
        smci_pre1.set_xlabel('Estimated SM(m³/m³)')
        smci_pre1.plot([0, 1], [0, 1], c='#d08a7d', lw=1)
        smci_pre1.set_title('{}'.format(list(smci_pre_list.keys())[i]))

    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()


def variate_plot(data, year, save_name):
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    data = reshape(data)[0, :, 0:15]
    data = np.array(data.cpu().detach().numpy())
    days_of_year = np.arange(1, 367)  # 一年中的日期，从1到366
    # 创建一个新的 Matplotlib 图形

    # 绘制折线图
    colors = ["#808AB1", '#a0a89b', '#b3ac75', '#408390']
    labels = ['Total_precipitation_sum', 'Volumetric_soil_water_layer_2_min',
              'Volumetric_soil_water_layer_1_max',
              'Lai', 'Leaf_area_index_high_vegetation_min',
              'Leaf_area_index_low_vegetation_min', 'Temperature_2m',
              'Leaf_area_index_high_vegetation_max',
              'Leaf_area_index_low_vegetation_max', 'Volumetric_soil_water_layer_2_max',
              'Volumetric_soil_water_layer_1_min', 'Total_evaporation_sum',
              'Potential_evaporation_sum',
              'Week_accumulation_p', 'Month_accumulation_p']

    for num in range(data.shape[1]):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_ylabel("Normalized Value")
        ax.set_xlabel("Day of Year {}".format(year))
        data1 = data[:, num].flatten()
        ax.plot(days_of_year, data1, linestyle="-", color='#408390')

        # 添加标题和标签
        plt.title(labels[num])

        # 添加图例
        # ax.legend(loc='upper right')

        # 显示图形
        plt.grid(True)
        # plt.savefig(save_name + '/{}.png'.format(labels[num]), bbox_inches='tight', dpi=300)
        # plt.show()
        plt.close()


def reshape(data):
    data = data.clone().detach().requires_grad_(True)
    data = torch.transpose(data, 1, 3)
    data = torch.transpose(data, 2, 4)
    data = data.reshape(data.shape[0] * data.shape[1] * data.shape[2], data.shape[3], data.shape[4])
    return data


if __name__ == "__main__":
    # 预测
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, default='2012')
    parser.add_argument("--depth", type=str, default='10cm')
    parser.add_argument("--ISMN", type=str, default='')  # 'ISMN_'
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--inputs", type=int, default=32)
    parser.add_argument("--outputs", type=int, default=1)
    parser.add_argument("--model_name", type=str, default='')  # 'LSTM', 'mamba', 'unet', 'vgg', 'mamba2'
    args = parser.parse_args()

    txt = '{}_{}_predictISMN_names.txt'.format(args.year, args.depth)

    with open(txt, "r") as f:
        total_names = f.readlines()
    images = [i[:-1] for i in total_names]

    # sample_size = int(len(total_names) * 0.05)
    # images = random.sample(total_names, sample_size)

    ismndict = np.load('tiles/ISMN/ISMNdict.npy', allow_pickle=True).item()
    ismn_path = 'tiles/ISMN/Data_separate_files_header_20000101_20201231_11023_ymr1_20240628.zip'
    ismn_data = ISMN_Interface(ismn_path, parallel=True)

    all_SM_list = {'ISMN': [], 'LSTM_ISMN': [], 'mamba_ISMN': [],
                   'SMCI1.0': [], 'LSTM_SMCI1.0': [], 'mamba_SMCI1.0': []}

    for image_name in images:
        model_names = ['LSTM', 'mamba']  # 'LSTM', 'mamba', 'unet', 'vgg'

        # for model_name in model_names:
        #
        #     parser.set_defaults(model_name=model_name)
        #     args = parser.parse_args()
        #
        #     image_path = 'tiles/{}_stack/{}/{}.tif'.format(args.year, args.depth, image_name)
        #     image = torch.tensor(gdal.Open(image_path).ReadAsArray())
        #     mean = np.load('mean_and_std/{}_{}_mean_bands.npy'.format(args.year, args.depth))
        #     std = np.load('mean_and_std/{}_{}_std_bands.npy'.format(args.year, args.depth))
        #     image[image == -999] = 0
        #
        #     image = inputs_reshape(image, normalize=True, mean=mean, std=std)
        #     image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2], image.shape[3])
        #     image = torch.tensor(image).to(device).float()
        #
        #     # 画变量
        #     # variate_save_name = 'train_and_predict_out/figure/variates/{}/{}/{}'.format(args.year, args.depth, image_name)
        #     # variate_plot(image, args.year, variate_save_name)
        #
        #     model = load_model(args).to(device)
        #     model_path = 'train_and_predict_out/model_pth/{}/{}/{}_{}best_weights.pth'.format(args.year, args.depth, model_name, args.ISMN)
        #     model.load_state_dict(torch.load(model_path))
        #     image_output = model(image)
        #     save_path = 'train_and_predict_out/prediction/{}/{}/{}{}_{}.npy'.format(args.year, args.depth, args.ISMN, model_name, image_name)
        #     np.save(save_path, image_output.cpu().detach().numpy())

        predictions = {}
        for model_name in model_names:
            parser.set_defaults(model_name=model_name)
            args = parser.parse_args()
            save_path = 'train_and_predict_out/prediction/{}/{}/{}{}_{}.npy'.format(args.year, args.depth, args.ISMN, model_name, image_name)
            mean_and_std_for_SM = np.load('norm_for_SM/{}_{}_SM_mean_and_std.npy'.format(args.year, args.depth))
            prediction = np.load(save_path) * mean_and_std_for_SM[1] + mean_and_std_for_SM[0]
            predictions[model_name] = prediction

        # SMCI1.0
        # SMCI_path = 'Soil_Moisture/SMCI_1km_{}_{}/{}.nc'.format(args.year, args.depth, image_name)
        SMCI_path = r'J:\research\soil_moistur_retrieval\SMCI_1km\10cm\SMCI_1km_{}_{}\{}.nc'.format(args.year, args.depth, image_name)
        SMCI_data = nc.Dataset(SMCI_path, mode='r').variables['SMCI']

        precipitaion_path = 'tiles/{}/{}/total_precipitation_sum/{}.tif'.format(args.year, args.depth, image_name)
        precipitaions = gdal.Open(precipitaion_path).ReadAsArray()

        ISMN_image = 'tiles/ismn_stations/max/{}.tif'.format(image_name)
        ISMN_image = torch.tensor(gdal.Open(ISMN_image).ReadAsArray())

        total_ismn = []
        total_ismn_day = []
        total_scmi = []
        total_pre = []

        for i in range(ISMN_image.shape[0]):
            for j in range(ISMN_image.shape[1]):
                if ISMN_image[i, j] == 0:
                    continue
                num_station = int(ISMN_image[i, j])
                ismn_station_info = ismndict.get(num_station)
                ismn_station_name = ismn_station_info.get('station')
                time_format = "%Y-%m-%d %H:%M:%S"
                ismn_station_timerange_from = datetime.strptime(ismn_station_info.get('timerange_from'), time_format)
                ismn_station_timerange_to = datetime.strptime(ismn_station_info.get('timerange_to'), time_format)
                if (ismn_station_timerange_from > datetime.strptime("{}-12-31 00:00:00".format(args.year), time_format) or
                    ismn_station_timerange_to < datetime.strptime("{}-01-01 00:00:00".format(args.year), time_format)):
                    continue
                ismn_station_data = ismn_data.network_for_station(ismn_station_name, name_only=False) \
                                              [ismn_station_name]
                ismn_station_sensor_names = list(ismn_station_data.sensors.keys())
                for sensor in ismn_station_sensor_names:
                    sensor_depth = float(sensor.split('_')[-1])
                    depth = float(args.depth[0:2])/100
                    if sensor_depth != depth:
                        continue
                    SMys = {}
                    SMxs = {}
                    ismn_sensor_data = ismn_station_data[sensor].data['soil_moisture'].resample('D').mean()
                    ismn_sensor_data = ismn_sensor_data['{}-01-01'.format(args.year):'{}-12-31'.format(args.year)]
                    ismn_sensor_y = ismn_sensor_data.to_numpy()
                    start = datetime.strptime('{}-01-01 00:00:00'.format(args.year), "%Y-%m-%d %H:%M:%S")
                    ismn_sensor_x = (ismn_sensor_data.index-start).days + 1
                    if len(ismn_sensor_x) == 0:
                        continue
                    if np.isnan(ismn_sensor_y).all():
                        continue
                    SMxs['ISMN-{}-{}'.format(ismn_station_name, sensor)] = ismn_sensor_x
                    SMys['ISMN-{}-{}'.format(ismn_station_name, sensor)] = ismn_sensor_y

                    SMxs['SMCI1.0'] = np.arange(1, 367)
                    SMys['SMCI1.0'] = SMCI_data[:, i, j].flatten() / 1000

                    ismn_pre_list = {}
                    smci_pre_list = {}
                    for model_name in model_names:
                        SMxs[model_name] = np.arange(1, 367)
                        SMys[model_name] = predictions[model_name][:, :, i, j].flatten()/1000

                        # ubrmse, bias, mae, r, r2, mse, pre, ismn
                        ismn_pre_list[model_name] = ismn_pre_loss(ismn_sensor_y, ismn_sensor_x, SMys[model_name])
                        if model_name == 'LSTM':
                            all_SM_list['ISMN'] = all_SM_list['ISMN'] + ismn_pre_list[model_name][7]
                        all_SM_list[model_name +'_ISMN'] = all_SM_list[model_name +'_ISMN'] + \
                                                           ismn_pre_list[model_name][6]

                        # ubrmse, bias, mae, r, r2, mse, pre, smci
                        smci_pre_list[model_name] = smci_pre_loss(SMys['SMCI1.0'], SMys[model_name])
                        if model_name == 'LSTM':
                            all_SM_list['SMCI1.0'] = all_SM_list['SMCI1.0'] + smci_pre_list[model_name][7]
                        all_SM_list[model_name + '_SMCI1.0'] = all_SM_list[model_name + '_SMCI1.0'] + \
                                                               smci_pre_list[model_name][6]
                    # 画图
                    precipitaion = precipitaions[:, i, j].flatten()

                    save_name = 'train_and_predict_out/figure/{}/{}/{}{}_{}_{}_{}_{}_best_SMcomparasion.png'.format(
                        args.year,
                        args.depth, args.ISMN,
                        image_name, i, j,
                        ismn_station_name,
                        sensor)

                    SM_comparasion(SMys, SMxs, precipitaion * 1000, args.year, save_name, image_name, ismn_pre_list, smci_pre_list)
    save_name = 'train_and_predict_out/figure/{}/{}/{}all_ISMNstation_best_SMcomparasion.png'.format(args.year, args.depth, args.ISMN)
    SM_scatter_comparasion(all_SM_list, save_name)
