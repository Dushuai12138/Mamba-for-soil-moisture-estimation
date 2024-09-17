# -*- coding: utf-8 -*-
# time: 2024/7/4 16:44
# file: loss.py
# author: Shuai


import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=str, default='2012')
parser.add_argument("--depth", type=str, default='10cm')
args = parser.parse_args()

for mode in ['train', 'val']:
    save_name = r'H:\soil_moistur_retrieval\to_chaosuan\loss\figure\{}_{}_{}_loss.png'.format(args.year, args.depth,
                                                                                              mode)
    for model in ['mamba', 'LSTM']:
        file = r'H:\soil_moistur_retrieval\to_chaosuan\loss\{}_{}_{}_{}_loss.txt'.format(args.year, args.depth, model, mode)

        # 读取 loss.txt 文件
        x = 1
        xs = []
        with open(file, 'r') as file:
            lines = file.readlines()
            losses = []
            for line in lines:
                loss = float(line.strip())
                losses.append(loss)
                xs.append(x)
                x = x + 1

        # 绘制损失函数图
        plt.plot(xs, losses, label=model)
    plt.xlabel('Epoch')
    plt.ylabel('{} Loss'.format(mode))
    plt.legend()
    # plt.title('Loss Function')
    plt.grid(True)
    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.show()
