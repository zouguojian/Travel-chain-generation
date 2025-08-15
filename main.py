import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim

import numpy as np
import json
from time import time
from datetime import datetime
import argparse
import random

import sys
import os

from model import DMTLN

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--test", action="store_true", help="test program")
parser.add_argument('--gcn_bool', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--aptonly', type=bool, default=False, help='whether only adaptive adj')
parser.add_argument('--addaptadj', type=bool, default=True, help='whether add adaptive adj')
parser.add_argument('--randomadj', type=bool, default=True, help='whether random initialize adaptive adj')

args = parser.parse_args()
config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
print(json.dumps(config, sort_keys=True, indent=4))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

north_south_map = config['north_south_map']
west_east_map = config['west_east_map']

all_data_filename = config['all_data_filename']    # (8760, 48, 20, 20)
mask_filename = config['mask_filename']            # (20, 20)
road_adj_filename = config['road_adj_filename']    # (243, 243)
risk_adj_filename = config['risk_adj_filename']    # (243, 243)
poi_adj_filename = config['poi_adj_filename']      # (243, 243)
grid_node_filename = config['grid_node_filename']  # (400, 243)
patience = config['patience']
delta = config['delta']

if config['seed'] is not None:
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

'''
基础参数，包括数据的划分比率、数据特征尺度、epoch等信息
'''
train_rate = config['train_rate']
valid_rate = config['valid_rate']
recent_prior = config['recent_prior']
week_prior = config['week_prior']
one_day_period = config['one_day_period']
days_of_week = config['days_of_week']
pre_len = config['pre_len']
seq_len = recent_prior + week_prior  # 一条数据样本的时间步长
training_epoch = config['training_epoch']


def train(name):
    train_urdu = 'I am a student'
    train_eng = '我是学生'
    MAX_LENGTH = max(len(sentence.split()) for sentence in train_urdu + train_eng)
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

    a = {2:4, 4:5}
    print(len(a))

def main(config):
    criterion = nn.NLLLoss(ignore_index=PAD_token)
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(config)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
