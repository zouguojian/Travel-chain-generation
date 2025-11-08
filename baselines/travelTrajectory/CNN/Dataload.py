import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
class VariableLengthDataset(Dataset):
    def __init__(self, x, l):
        # 构建x：城市信息、车牌、车型、day of week、minute of day、路线、距离列表
        # x 表示模型的输入数据 [B, D]
        # y 表示标签 ([B, L], [B, 1])
        # l 表示每个路径的长度 [B, 1]
        # Initialize dataset with variable-length sequences
        self.data = []
        self.classification = []
        self.seq_lengths = []
        for i in range(l.shape[0]):
            # variable sequence length for x and y
            self.data.append(torch.FloatTensor(np.array(x[i][:-2], dtype=np.float32)))
            self.classification.append(torch.IntTensor(np.array([x[i][5]], dtype=np.int32))) # 路径, 代表类型
            self.seq_lengths.append(l[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return sequence, label, and sequence length
        # x: [0, 0, 13, 0, 1431, 1, 14388.9, 14828.39], y: [25.567, 11.9, 13.667], l: [2]
        return self.data[idx], self.classification[idx], self.seq_lengths[idx]


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

    data, classification, seq_lengths = zip(*batch)  # 分离数据: [B, R, L, D]、标签: [B, 1 + L]、长度: [B]
    # 初始化填充张量，数据用0填充，标签用-1填充以标记无效位置
    padded_x = torch.zeros(len(batch), data[0].shape[0]) # [B, 5]
    padded_classification = torch.zeros(len(batch), 1, dtype=torch.long)
    # 将每个样本填充到最大长度
    for i in range(len(batch)):
        padded_x[i] = data[i]
        padded_classification[i] = classification[i]
    return padded_x, padded_classification, seq_lengths

def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    data = {}
    max_dow, max_mod= 7, 1440
    # 加载数据到data字典中
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'), allow_pickle=True)
        data['x_' + category] = cat_data['x']
        data['l_' + category] = cat_data['l']
        del cat_data

    total_x = np.concatenate((data['x_' + 'train'], data['x_' + 'val'], data['x_' + 'test']))
    max_city, max_plate, max_v_type= np.max(total_x[:, 0]) + 1, np.max(total_x[:, 1]) + 1, np.max(total_x[:, 2]) + 1
    del total_x

    train_dataset = VariableLengthDataset(data['x_train'], data['l_train'])
    print('load train dataset done')
    val_dataset = VariableLengthDataset(data['x_val'], data['l_val'])
    print('load val dataset done')
    test_dataset = VariableLengthDataset(data['x_test'], data['l_test'])
    print('load test dataset done')

    # Generate synthetic variable-length dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, max_city, max_plate, max_v_type, max_dow, max_mod
    ''' 
    # for train_loader
    torch.Size([32, 5]) torch.Size([32, 1])
    '''

# train_loader, val_loader, test_loader, max_city, max_plate, max_v_type, max_dow, max_mod = load_dataset('/Users/zouguojian/Travel-chain-generation/data', 32, test_batch_size=1)
# for batch_x, batch_dow, batch_mod, batch_cla, batch_l in train_loader:
#     print(batch_x.shape, batch_dow.shape, batch_mod.shape, batch_cla.shape)



'''
torch.Size([32, 10, 8, 7]) torch.Size([32, 12, 66]) torch.Size([32, 12, 108]) torch.Size([32, 12, 1]) torch.Size([32, 12, 1]) torch.Size([32, 7]) torch.Size([32, 1])
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