# -*- coding: utf-8 -*-
# time: 2024/5/31 15:26
# file: make_dataset_txt.py
# author: Shuai

import os
import argparse


def for_train(label_folder, args, ISMN=False):
    names = os.listdir(label_folder)
    total_names = []
    for name in names:
        file_type = '.nc'
        if ISMN:
            file_type = '.tif'
        if name.endswith(file_type):
            total_names.append(name)
    txt = '{}_{}_names.txt'.format(args.year, args.depth)
    if ISMN:
        txt = '{}_{}_predictISMN_names.txt'.format(args.year, args.depth)
    names_file = open(txt, 'w')
    for i in total_names:
        i = i.split('.')[0]+'\n'
        names_file.write(i)
    names_file.close()


def for_predict():
    output_folder = r'H:\soil_moistur_retrieval\images_name'
    label_folder = r'H:\soil_moistur_retrieval\tiles\2020\lai'

    names = os.listdir(label_folder)
    total_names = []
    for name in names:
        if name.endswith(".tif"):
            total_names.append(name)

    names_file = open(os.path.join(output_folder, 'predict_names.txt'), 'w')
    for i in total_names:
        i = i[:-4] + '\n'
        names_file.write(i)
    names_file.close()


parser = argparse.ArgumentParser()
parser.add_argument("--year", type=str, default='2012')
parser.add_argument("--depth", type=str, default='10cm')
args = parser.parse_args()
label_folder = 'Soil_Moisture/SMCI_1km_2012_10cm'
# label_folder = r'H:\soil_moistur_retrieval\to_chaosuan\tiles\ismn_stations\max'

for_train(label_folder, args, ISMN=False)
