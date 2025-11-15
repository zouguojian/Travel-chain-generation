import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd

# 地理邻接矩阵和语义邻接矩阵
def load_adjs(geo_adj_fil = 'data/states/geo_flow_adj.npz', sem_adj_fil = 'data/states/sem_flow_adj.npz', device = 'cpu'):
    # 返回torch类型的张量
    geo_adj = np.load(geo_adj_fil, allow_pickle=True)['data']
    geo_adj = torch.from_numpy(np.array(geo_adj, dtype=np.float32)).to(device=device) # 地理空间

    sem_adj = np.load(sem_adj_fil, allow_pickle=True)['data']
    sem_adj = torch.from_numpy(np.array(sem_adj, dtype=np.float32)).to(device=device) # 语义空间

    return geo_adj, sem_adj

# 每一条路段的index
def load_segment_index(segment_indx_fil = 'data/path_segment_ids.csv'):
    '''
        [array([80], dtype=int32) array([28, 30], dtype=int32)]
    '''
    df = pd.read_csv(segment_indx_fil, encoding='utf-8')
    lengths = df['Length']
    segment_indices = [item[:lengths[i]].astype(np.int32) for i, item in enumerate(df.values[:, 2:])]
    segment_indices = np.array(segment_indices, dtype=object)
    return segment_indices

# 每一条路段的的长度
def load_segment_distance(segment_dis_fil = 'data/states/segment_station_with_distance.csv'):
    '''
        [array([1.25489839]) array([1.41415507, 1.49492749])]
    '''
    segment_indices = load_segment_index() # 用于获取对应路段的index
    dis = pd.read_csv(segment_dis_fil, usecols=['distance']).values.reshape(-1)
    mean, std = np.mean(dis), np.std(dis)
    dis = np.array([(dis[item] - mean) / std for item in segment_indices], dtype=object)
    return dis # 归一化的数值

class VariableLengthDataset(Dataset):
    def __init__(self, x, speed, y, l, mean, std):
        # x 表示模型的输入数据 [B, D]
        # y 表示标签 ([B, L], [B, 1])
        # l 表示每个路径的长度 [B, 1]
        # speed流量数据 [B, T, N, 1]
        # Initialize dataset with variable-length sequences
        self.data = []
        self.speed_data = []
        self.labels = []
        self.seq_lengths = []
        dis = load_segment_distance()  # 记录的是每条路线上的路段长度
        indices = load_segment_index()
        for i in range(l.shape[0]):
            # variable sequence length for x and y
            x_1 = np.array(np.tile(x[i][:-2], (dis[x[i][5]].shape[0], 1)), dtype=np.float32)
            x_2 = np.array(np.reshape(indices[x[i][5]], [indices[x[i][5]].shape[0], 1]), dtype=np.int32)
            x_3 = np.array(np.reshape(dis[x[i][5]], [dis[x[i][5]].shape[0], 1]), dtype=np.float32)
            x_4 = np.concatenate((x_1, x_2, x_3), axis=1) # [L, D] D = 7
            # X：城市信息、车牌、车型、day of week、minute of day、路段编号、路段长度
            self.data.append(torch.FloatTensor(x_4))
            self.speed_data.append(torch.FloatTensor(np.reshape(speed[i,-1,indices[x[i][5]]], (dis[x[i][5]].shape[0], 1))))
            self.labels.append(torch.FloatTensor(np.array([y[i][1]] + y[i][0], dtype=np.float32)))
            self.seq_lengths.append(l[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return sequence, label, and sequence length
        # x: [0, 0, 13, 0, 1431, 1, 14388.9, 14828.39], y: [25.567, 11.9, 13.667], l: [2]
        return self.data[idx], self.speed_data[idx], self.labels[idx], self.seq_lengths[idx]

def X_paded(X, max_l):
    '''
    # 主要用于填充路线
    X is list, element is [L, D], L是变化的
    '''
    padded_x = torch.zeros(len(X), max_l, X[0].shape[1])  # [B, max_l, 6 + 1]
    for i in range(len(X)):
        padded_x[i, :X[i].shape[0]] = X[i]
    return padded_x

# 自定义collate_fn，用于动态填充变长序列以形成批次
def collate_fn(batch):
    """
    将变长序列样本填充到统一长度，形成批次张量。

    参数:
        batch: list of (data, label) tuples, where data is [L, D] and label is [L, 1].

    返回:
        padded_x: [B, max_l, D], 填充后的输入数据张量，填充值为0。
        padded_y: [B, max_l, 1], 填充后的标签张量，填充值为-1（用于掩码）。值得注意，第一个维度是总时间，后面是路段时间
    """
    # Pad sequences to the longest in the batch
    # Input: batch (list of (sequence, label, seq_length))
    # Output: padded_data [B, max_len, D], labels [B], seq_lengths (list)

    data, speed, labels, seq_lengths = zip(*batch)  # 分离数据: [B, L, D]、标签: [B, 1 + L]、长度: [B]
    max_l = max(seq_lengths)  # 批次中的最大序列长度
    # 初始化填充张量，数据用0填充，标签用-1填充以标记无效位置
    padded_x = X_paded(data, max_l)
    # padded_x = torch.zeros(len(batch), data[0].shape[0], data[0].shape[1]) # [B, max_l, 6 + 1]
    padded_y = torch.ones(len(batch), max_l + 1) * -1  # -1表示填充位置 [B, 1 + max_l]
    # padded_speed = torch.zeros(len(batch), speed[0].shape[0], speed[0].shape[1])
    padded_speed = X_paded(speed, max_l)
    # 将每个样本填充到最大长度
    for i in range(len(batch)):
        # padded_x[i] = data[i]
        padded_y[i, :seq_lengths[i] + 1] = labels[i]
        # padded_speed[i] = speed[i]
    return padded_x, padded_speed, padded_y, seq_lengths

def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    data = {}
    max_dow, max_mod= 7, 1440
    # 加载数据到data字典中
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'), allow_pickle=True)
        data['x_' + category] = cat_data['x']
        data['speed_x_' + category] = cat_data['speed_x']
        data['y_' + category] = cat_data['y']
        data['l_' + category] = cat_data['l']
        xdis_mean, xdis_std = cat_data['xdis_mean'], cat_data['xdis_std']
        ytra_mean, ytra_std = cat_data['ytra_mean'], cat_data['ytra_std']
        ytol_mean, ytol_std = cat_data['ytol_mean'], cat_data['ytol_std']
        del cat_data

    total_x = np.concatenate((data['x_' + 'train'], data['x_' + 'val'], data['x_' + 'test']))
    max_city, max_plate, max_v_type = np.max(total_x[:, 0]) + 1, np.max(total_x[:, 1]) + 1, np.max(total_x[:, 2]) + 1
    del total_x

    train_dataset = VariableLengthDataset(data['x_train'],  data['speed_x_train'], data['y_train'], data['l_train'], xdis_mean, xdis_std)
    print('load train dataset done')
    val_dataset = VariableLengthDataset(data['x_val'], data['speed_x_val'], data['y_val'], data['l_val'], xdis_mean, xdis_std)
    print('load val dataset done')
    test_dataset = VariableLengthDataset(data['x_test'], data['speed_x_test'], data['y_test'], data['l_test'], xdis_mean, xdis_std)
    print('load test dataset done')

    # Generate synthetic variable-length dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, max_city, max_plate, max_v_type, max_dow, max_mod, ytra_mean.item(), ytra_std.item(), ytol_mean.item(), ytol_std.item()
    '''
    torch.Size([32, 8, 7]) torch.Size([32, 8, 1]) torch.Size([32, 9])
    '''


# train_loader, val_loader, test_loader, max_city, max_plate, max_v_type, max_dow, max_mod, ytra_mean, ytra_std, ytol_mean, ytol_std = load_dataset('/Users/zouguojian/Travel-chain-generation/data', 32, test_batch_size=1)
# for batch_x, batch_speed, batch_y, batch_l in train_loader:
#     print(batch_x.shape, batch_speed.shape, batch_y.shape)