# -*- coding: utf-8 -*-
# time: 2024/7/16 14:36
# file: utils.py
# author: Shuai

from datetime import datetime
import torch
import torch.nn as nn
from osgeo import gdal
import numpy as np
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ismn_loss(image_name, prediction, ismndict, ismn_data, args):
    '''
    :param image_name: like '46_89'
    :param prediction: array with dimensions of [366, 64, 64]
    :return: loss
    '''
    ismn_loss_toatl_value = 0
    criterion = nn.MSELoss(reduction='mean').to(device)
    ISMN_image = 'tiles/ismn_stations/max/{}.tif'.format(image_name)
    ISMN_image = torch.tensor(gdal.Open(ISMN_image).ReadAsArray())
    for i in range(ISMN_image.shape[0]):
        for j in range(ISMN_image.shape[1]):
            if ISMN_image[i, j] == 0:
                continue
            num_station = int(ISMN_image[i, j])
            ismn_station_info = ismndict.get(num_station)
            print(ismn_station_info)
            ismn_station_name = ismn_station_info.get('station')
            time_format = "%Y-%m-%d %H:%M:%S"
            ismn_station_timerange_from = datetime.strptime(ismn_station_info.get('timerange_from'), time_format)
            ismn_station_timerange_to = datetime.strptime(ismn_station_info.get('timerange_to'), time_format)
            if (ismn_station_timerange_from > datetime.strptime("{}-12-31 00:00:00".format(args.year), time_format) or
                    ismn_station_timerange_to < datetime.strptime("{}-01-01 00:00:00".format(args.year), time_format)):
                continue
            ismn_station_data = ismn_data.network_for_station(ismn_station_name, name_only=False)[ismn_station_name]
            ismn_station_sensor_names = list(ismn_station_data.sensors.keys())
            for sensor in ismn_station_sensor_names:
                sensor_depth = float(sensor.split('_')[-1])
                depth = float(args.depth[0:2]) / 100
                if sensor_depth != depth:
                    continue
                ismn_sensor_data = ismn_station_data[sensor].data['soil_moisture'].resample('D').mean()
                ismn_sensor_data = ismn_sensor_data['{}-01-01'.format(args.year):'{}-12-31'.format(args.year)]
                ismn_sensor_y = torch.tensor(ismn_sensor_data.to_numpy())
                start = datetime.strptime('{}-01-01 00:00:00'.format(args.year), "%Y-%m-%d %H:%M:%S")
                ismn_sensor_x = (ismn_sensor_data.index - start).days
                if len(ismn_sensor_x) == 0:
                    continue
                valid_indices = ~torch.isnan(ismn_sensor_y)
                ismn_sensor_x = [ismn_sensor_x[position] for position in range(len(ismn_sensor_y)) if valid_indices[position]]
                ismn_sensor_y = ismn_sensor_y[valid_indices]

                prediction_position = prediction[:, i, j].flatten()

                prediction_sm = torch.tensor([prediction_position[position] for position in ismn_sensor_x])
                ismn_loss_value = criterion(prediction_sm, ismn_sensor_y)
                ismn_loss_value = torch.where(torch.isnan(ismn_loss_value), torch.tensor(0), ismn_loss_value)

                ismn_loss_toatl_value = ismn_loss_toatl_value + ismn_loss_value

    return ismn_loss_toatl_value


def calculate_loss(predicted, measured):
    """
    计算预测值和测量值之间的无偏均方根误差（UBRMSE）

    参数：
    predicted (numpy array): 预测值数组
    measured (numpy array): 测量值数组

    返回值：
    float: 预测值和测量值之间的UBRMSE
    """
    # 检查预测值和测量值的长度是否一致
    if len(predicted) != len(measured):
        raise ValueError("预测值和测量值数组的长度必须一致")

    # 计算预测值和测量值的平均值
    mean_predicted = np.mean(predicted)
    mean_measured = np.mean(measured)

    # 计算无偏差的预测值和测量值
    unbiased_predicted = predicted - mean_predicted
    unbiased_measured = measured - mean_measured

    # 计算无偏均方根误差（UBRMSE）
    ubrmse = np.sqrt(np.mean((unbiased_predicted - unbiased_measured) ** 2))
    bias = np.mean(predicted - measured)
    mae = np.mean(np.abs(predicted - measured))
    r = np.sum(unbiased_predicted * unbiased_measured) / np.sqrt(np.sum(unbiased_predicted ** 2)) / np.sqrt(np.sum(unbiased_measured ** 2))
    mse = np.mean((measured - predicted) ** 2)
    r2 = 1 - np.sum((predicted - measured) ** 2)/np.sum((measured - mean_measured) ** 2)

    # r2 = r2_score(measured, predicted)

    return ubrmse, bias, mae, r, r2, mse


def ismn_pre_loss(ismn, ismn_day, pre):
    ismn_day = ismn_day - 1
    valid_indices = ~np.isnan(ismn)
    ismn_day = [ismn_day[position] for position in range(len(ismn)) if valid_indices[position]]
    ismn = ismn[valid_indices]

    pre = [pre[position] for position in ismn_day]
    ubrmse, bias, mae, r, r2, mse = calculate_loss(pre, ismn)
    return ubrmse, bias, mae, r, r2, mse, list(pre), list(ismn)


def smci_pre_loss(smci, pre):
    ubrmse, bias, mae, r, r2, mse = calculate_loss(pre, smci)
    return ubrmse, bias, mae, r, r2, mse, list(pre), list(smci)


def ismn_pixels_loss(ismn, outs, ismn_index, requires_grad):
    outs = outs.flatten()
    ismn = ismn.flatten()
    ismn_index = ismn_index.flatten()

    outs = np.array([outs[position].cpu().detach().numpy() if ismn_index[position] == 1 else 0 for position in range(len(ismn_index))])
    outs = outs[outs != 0]
    ismn = np.array([ismn[position].cpu().detach().numpy() if ismn_index[position] == 1 else 0 for position in range(len(ismn_index))])
    ismn = ismn[ismn != 0]

    ubrmse, bias, mae, r, r2, mse, pre, smci = smci_pre_loss(ismn, outs)

    return torch.tensor(mse, requires_grad=requires_grad)
