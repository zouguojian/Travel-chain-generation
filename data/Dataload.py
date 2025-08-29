import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class VariableLengthDataset(Dataset):
    def __init__(self, x, flow, speed, dow, mod, y, l, mean, std):
        # x 表示模型的输入数据 [B, D]
        # y 表示标签 ([B, L], [B, 1])
        # l 表示每个路径的长度 [B, 1]
        # Initialize dataset with variable-length sequences
        self.data = []
        self.flow_data = []
        self.speed_data = []
        self.dow_data = []
        self.mod_data = []
        self.labels = []
        self.seq_lengths = []
        for i in range(l.shape[0]):
            # variable sequence length for x and y
            x_1 = np.array(np.tile(x[i][:-1], (l[i], 1)), dtype=np.float32)
            x_2 = (np.array(np.reshape(x[i][-1], [l[i], 1]), dtype=np.float32) - mean) / std # 归一化
            self.data.append(torch.FloatTensor(np.concatenate((x_1, x_2), axis=1)))
            self.flow_data.append(torch.FloatTensor(flow[i]))
            self.speed_data.append(torch.FloatTensor(speed[i]))
            # print(type(dow[i]), dow[i].shape, type(mod[i]), mod[i].shape)
            self.dow_data.append(torch.IntTensor(dow[i]))
            self.mod_data.append(torch.IntTensor(mod[i]))
            self.labels.append(torch.FloatTensor(np.array([y[i][1]] + y[i][0], dtype=np.float32)))
            self.seq_lengths.append(l[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return sequence, label, and sequence length
        # x: [0, 0, 13, 0, 1431, 1, 14388.9, 14828.39], y: [25.567, 11.9, 13.667], l: [2]
        return self.data[idx], self.flow_data[idx], self.speed_data[idx], self.dow_data[idx], self.mod_data[idx], self.labels[idx], self.seq_lengths[idx]


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

    data, flow, speed, dow, mod, labels, seq_lengths = zip(*batch)  # 分离数据: [B, 6 + L]、标签: [B, 1 + L]、长度: [B]
    max_l = max(seq_lengths)  # 批次中的最大序列长度
    # 初始化填充张量，数据用0填充，标签用-1填充以标记无效位置
    padded_x = torch.zeros(len(batch), max_l, data[0].shape[1]) # [B, max_l, 6 + 1]
    padded_y = torch.ones(len(batch), max_l + 1) * -1  # -1表示填充位置 [B, 1 + max_l]
    padded_flow = torch.zeros(len(batch), flow[0].shape[0], flow[0].shape[1])
    padded_speed = torch.zeros(len(batch), speed[0].shape[0], speed[0].shape[1])
    padded_dow = torch.zeros(len(batch), dow[0].shape[0], dow[0].shape[1])
    padded_mod = torch.zeros(len(batch), mod[0].shape[0], mod[0].shape[1])
    # 将每个样本填充到最大长度
    for i in range(len(batch)):
        padded_x[i, :seq_lengths[i]] = data[i]
        padded_y[i, :seq_lengths[i] + 1] = labels[i]
        padded_flow[i] = flow[i]
        padded_speed[i] = speed[i]
        padded_dow[i] = dow[i]
        padded_mod[i] = mod[i]
    return padded_x, padded_flow, padded_speed, padded_dow, padded_mod, padded_y, seq_lengths

def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    data = {}
    max_dow, max_mod= 7, 1440
    # 加载数据到data字典中
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'), allow_pickle=True)
        data['x_' + category] = cat_data['x']
        data['flow_x_' + category] = cat_data['flow_x']
        data['speed_x_' + category] = cat_data['speed_x']
        data['dow_' + category] = cat_data['dow']
        data['mod_' + category] = cat_data['mod']
        data['y_' + category] = cat_data['y']
        data['l_' + category] = cat_data['l']
        xdis_mean, xdis_std = cat_data['xdis_mean'], cat_data['xdis_std']
        ytra_mean, ytra_std = cat_data['ytra_mean'], cat_data['ytra_std']
        ytol_mean, ytol_std = cat_data['ytol_mean'], cat_data['ytol_std']
        del cat_data

    total_x = np.concatenate((data['x_' + 'train'], data['x_' + 'val'], data['x_' + 'test']))
    max_city, max_plate, max_v_type, max_route = np.max(total_x[:, 0]) + 1, np.max(total_x[:, 1]) + 1, np.max(total_x[:, 2]) + 1, np.max(total_x[:, 5]) + 1
    del total_x

    train_dataset = VariableLengthDataset(data['x_train'], data['flow_x_train'], data['speed_x_train'], data['dow_train'], data['mod_train'], data['y_train'], data['l_train'], xdis_mean, xdis_std)
    print('load train dataset done')
    val_dataset = VariableLengthDataset(data['x_val'], data['flow_x_val'], data['speed_x_val'], data['dow_val'], data['mod_val'], data['y_val'], data['l_val'], xdis_mean, xdis_std)
    print('load val dataset done')
    test_dataset = VariableLengthDataset(data['x_test'], data['flow_x_test'], data['speed_x_test'], data['dow_test'], data['mod_test'], data['y_test'], data['l_test'], xdis_mean, xdis_std)
    print('load test dataset done')

    # Generate synthetic variable-length dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, max_city, max_plate, max_v_type, max_dow, max_mod, max_route, ytra_mean.item(), ytra_std.item(), ytol_mean.item(), ytol_std.item()
    ''' 
    # for train_loader
    torch.Size([32, 6, 7]) torch.Size([32, 12, 66]) torch.Size([32, 12, 108]) torch.Size([32, 12, 1]) torch.Size([32, 12, 1]) torch.Size([32, 7])
    '''


'''
train x:  (377803, 7) y: (377803, 2) flow: (377803, 12, 66) speed: (377803, 12, 108) dow: (377803, 12, 1) mod: (377803, 12, 1) l: (377803,)
'''

# train_loader, val_loader, test_loader, max_city, max_plate, max_v_type, max_dow, max_mod, max_rote, ytra_mean, ytra_std, ytol_mean, ytol_std = load_dataset('/Users/zouguojian/Travel-chain-generation/data', 32, test_batch_size=1)
# for batch_x, batch_flow, batch_speed, batch_dow, batch_mod, batch_y, batch_l in train_loader:
#     print(batch_x.shape, batch_flow.shape, batch_speed.shape, batch_dow.shape, batch_mod.shape, batch_y.shape)
'''
torch.Size([32, 6, 7]) torch.Size([32, 12, 66]) torch.Size([32, 12, 108]) torch.Size([32, 12, 1]) torch.Size([32, 12, 1]) torch.Size([32, 7])
'''

'''
data = {}
for category in ['train', 'val', 'test']:
    cat_data = np.load(os.path.join(category + '.npz'), allow_pickle=True)
    data['x_' + category] = cat_data['x']
    data['y_' + category] = cat_data['y']
    data['l_' + category] = cat_data['l']
    del cat_data

total_x = np.concatenate((data['x_' + 'train'], data['x_' + 'val'], data['x_' + 'test']))
print(total_x[:, -1])
x_mean, x_std = np.mean(np.reshape(total_x[:, -1], [-1])), np.std(np.reshape(total_x[:, -1], [-1]))
print(x_mean, x_std)

print(data['x_train'][-1], data['y_' + category][-1], data['l_' + category][-1])
train_dataset = VariableLengthDataset(data['x_train'], data['y_train'], data['l_train'])
print(train_dataset.__getitem__(-1))
'''


# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# for batch_x, batch_y, batch_l in train_loader:
#     print(batch_l)