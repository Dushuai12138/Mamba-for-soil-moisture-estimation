# -*- coding: utf-8 -*-
# time: 2024/5/31 15:27
# file: train.py
# author: Shuai


import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataloader import MyCustomDataset
from load_model import load_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import datetime
import random
import numpy as np
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import argparse
from utils import ismn_loss
from ismn.interface import ISMN_Interface
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, default='2012')
    parser.add_argument("--depth", type=str, default='10cm')
    parser.add_argument("--epoches", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--inputs", type=int, default=32)
    parser.add_argument("--outputs", type=int, default=1)
    parser.add_argument("--model_name", type=str)  # 'LSTM', 'mamba', 'unet', 'vgg', 'mamba2'
    parser.add_argument("--pretrained", type=str, default=False)
    parser.add_argument("--model_path", type=str, default='model_pth/mamba_30epoch_weights.pth')
    args = parser.parse_args()

    with open('{}_{}_names.txt'.format(args.year, args.depth), "r") as f:
        total_names = f.readlines()
    total_names = [i[:-1] for i in total_names]

    sample_size = int(len(total_names) * 0.001)
    total_names = random.sample(total_names, sample_size)

    names_file = open('{}_{}_used_names.txt'.format(args.year, args.depth), 'w')
    for i in total_names:
        i = i + '\n'
        names_file.write(i)
    names_file.close()

    train_input_names, test_input_names, train_labels_names, test_labels_names = train_test_split(total_names,
                                                                                                  total_names,
                                                                                                  test_size=0.3,
                                                                                                  random_state=42)

    train_num = len(train_input_names)
    test_num = len(test_input_names)

    mean = np.load('mean_and_std/{}_{}_mean_bands.npy'.format(args.year, args.depth))
    std = np.load('mean_and_std/{}_{}_std_bands.npy'.format(args.year, args.depth))
    mean_and_std_for_SM = np.load('norm_for_SM/{}_{}_SM_mean_and_std.npy'.format(args.year, args.depth))

    # 创建自定义数据集对象
    # bands = os.listdir(r'H:\soil_moistur_retrieval\tiles\2020')
    # SM_path = r'J:\research\soil_moistur_retrieval\SMCI_1km\10cm\SMCI_1km_{}_{}'.format(args.year, args.depth)
    bands = os.listdir('tiles/{}/{}'.format(args.year, args.depth))
    dataset_path = 'tiles/{}_stack/{}'.format(args.year, args.depth)
    SM_path = 'Soil_Moisture/SMCI_1km_{}_{}'.format(args.year, args.depth)
    train_dataset = MyCustomDataset(train_input_names, bands, dataset_path=dataset_path,
                                    SM_path=SM_path, mean=mean, std=std, mean_and_std_for_SM=mean_and_std_for_SM)
    test_dataset = MyCustomDataset(test_input_names, bands, dataset_path=dataset_path,
                                   SM_path=SM_path, mean=mean, std=std, mean_and_std_for_SM=mean_and_std_for_SM)

    # 创建 DataLoader 对象
    batch_size = args.batch_size

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model_name = args.model_name
    print(model_name)
    # 创建模型
    model = load_model(args=args).to(device)
    pretrained = args.pretrained
    if pretrained:
        model_path = args.model_path
        model.load_state_dict(torch.load(model_path))
        print('Load pre-trained model successfully!')

    current_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    with open('train_and_predict_out/time/{}_{}_{}_time.txt'.format(args.year, args.depth, str(model_name)), 'w') as f:
        f.write('start time: {}\n'.format(str(current_time)))
        f.close()

    process = open('train_and_predict_out/process/{}_{}_{}_process.txt'.format(args.year, args.depth, str(model_name)), 'w')
    process.write("训练集大小：{}\n".format(str(len(train_input_names))))
    process.write("测试集大小：{}\n".format(str(len(test_input_names))))
    process.close()
    print("训练集大小:", len(train_input_names))
    print("测试集大小：", len(test_input_names))

    # 定义损失函数和优化器
    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # 定义学习率衰减策略（这里使用StepLR）
    # scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    scheduler = CosineAnnealingLR(optimizer, T_max=210)

    learningRate = open('train_and_predict_out/LR/{}_{}_{}_learningRate.txt'.format(args.year, args.depth, str(model_name)), 'w')
    learningRate.write("{}\n".format(optimizer.param_groups[0]['lr']))
    learningRate.close()

    # 记录损失
    trainLoss_file = open(os.path.join('train_and_predict_out/loss/{}_{}_{}_train_loss.txt'.format(args.year, args.depth, str(model_name))), 'w')
    trainLoss_file.close()

    valLoss_file = open(os.path.join('train_and_predict_out/loss/{}_{}_{}_val_loss.txt'.format(args.year, args.depth, str(model_name))), 'w')
    valLoss_file.close()

    # 遍历 DataLoader 对象
    val_loss_list = []
    train_loss_list = []
    epoches = args.epoches
    last_ten_loss = [1] * 10
    for epoch in range(0, epoches):
        current_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        with open('train_and_predict_out/time/{}_{}_{}_time.txt'.format(args.year, args.depth, str(model_name)), 'a') as f:
            f.write('opech: {}  start time: {}\n'.format(epoch, str(current_time)))
            f.close()

        total_train_loss = 0.0
        total_val_loss = 0.0

        model.train()
        for iteration, batch in enumerate(train_data_loader):
            train_input_data = batch['image'].to(device).float()
            train_labels = batch['label'].to(device).float()
            image_names = batch['image_name']
            optimizer.zero_grad()
            if args.model_name == 'transformer':
                outputs, train_labels = model(train_input_data, train_labels)
            else:
                outputs = model(train_input_data)
            loss = criterion(outputs, train_labels)
            loss = torch.sqrt(loss)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            process = open('train_and_predict_out/process/{}_{}_{}_process.txt'.format(args.year, args.depth, str(model_name)), 'a')
            process.write("iteration: {}, epoch: {}, Train Total Loss: {}\n".format(iteration, epoch, total_train_loss))
            process.close()
            print("iteration: %s, epoch: %s, Train Total Loss: %s" % (iteration, epoch, total_train_loss))

        # 调整学习率
        scheduler.step()

        average_train_loss = total_train_loss / len(train_data_loader)
        trainLoss_file = open(os.path.join('train_and_predict_out/loss/{}_{}_{}_train_loss.txt'.format(args.year, args.depth, str(model_name))), 'a')
        trainLoss_file.write(str(average_train_loss) + '\n')
        trainLoss_file.close()

        learningRate = open('train_and_predict_out/LR/{}_{}_{}_learningRate.txt'.format(args.year, args.depth, str(model_name)), 'a')
        learningRate.write("{}\n".format(optimizer.param_groups[0]['lr']))
        learningRate.close()

        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(test_data_loader):
                test_input_data = batch['image'].to(device).float()
                test_labels = batch['label'].to(device).float()
                image_names = batch['image_name']

                outputs = model(test_input_data)
                loss = criterion(outputs, test_labels)
                loss = torch.sqrt(loss)
                total_val_loss += loss.item()
                process = open('train_and_predict_out/process/{}_{}_{}_process.txt'.format(args.year, args.depth, str(model_name)), 'a')
                process.write("iteration: {}, epoch: {}, Val Total Loss: {}\n".format(iteration, epoch, total_val_loss))
                process.close()
                print("iteration: %s, epoch: %s, Val Total Loss: %s" % (iteration, epoch, total_val_loss))

        average_val_loss = total_val_loss / len(test_data_loader)
        valLoss_file = open(os.path.join('train_and_predict_out/loss/{}_{}_{}_val_loss.txt'.format(args.year, args.depth, str(model_name))), 'a')
        valLoss_file.write(str(average_val_loss) + '\n')
        valLoss_file.close()

        process = open('train_and_predict_out/process/{}_{}_{}_process.txt'.format(args.year, args.depth, str(model_name)), 'a')
        process.write("epoch: {}, Average Loss: {}\n".format(epoch, average_train_loss))
        process.close()
        print("-------------     epoch: %s, Average Loss: %s     ------------" % (epoch, average_train_loss))

        last_ten_loss.append(average_val_loss)
        last_ten_loss = last_ten_loss[1:]
        average_last_ten_loss = np.mean(last_ten_loss)

        if average_last_ten_loss < average_val_loss and epoch >= 10:
            torch.save(model.state_dict(), 'train_and_predict_out/model_pth/{}/{}/{}_best_weights.pth'.format(args.year, args.depth, str(model_name)))
            print(str(average_last_ten_loss) + '<' + str(average_val_loss) + ':True')
            break

        if (epoch+1) % 10 == 0 and epoch != 0:
            # 保存模型
            torch.save(model.state_dict(), 'train_and_predict_out/model_pth/{}/{}/{}_{}epoch_weights.pth'.format(args.year, args.depth, str(model_name), epoch+1))

    current_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    with open('train_and_predict_out/time/{}_{}_{}_time.txt'.format(args.year, args.depth, str(model_name)), 'a') as f:
        f.write('end time: {}\n'.format(str(current_time)))
        f.close()
