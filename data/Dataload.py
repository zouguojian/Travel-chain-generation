import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


# 自定义数据集类，用于处理变长序列数据
class DeepFMDataset(Dataset):
    def __init__(self, data_list, labels_list):
        """
        初始化数据集，存储变长序列的输入数据和标签。

        参数:
            data_list: list of [L, D] tensors, where L is the variable sequence length
                       and D is the feature dimension (num_features).
            labels_list: list of [L, 1] tensors, containing binary labels (0 or 1).
        """
        self.data_list = data_list  # 存储变长输入数据列表
        self.labels_list = labels_list  # 存储变长标签列表

    def __len__(self):
        """返回数据集的样本总数"""
        return len(self.data_list)

    def __getitem__(self, idx):
        """获取指定索引的样本，返回数据和标签对"""
        return self.data_list[idx], self.labels_list[idx]


# 自定义collate_fn，用于动态填充变长序列以形成批次
def collate_fn(batch):
    """
    将变长序列样本填充到统一长度，形成批次张量。

    参数:
        batch: list of (data, label) tuples, where data is [L, D] and label is [L, 1].

    返回:
        padded_x: [B, max_l, D], 填充后的输入数据张量，填充值为0。
        padded_y: [B, max_l, 1], 填充后的标签张量，填充值为-1（用于掩码）。
    """
    x, y = zip(*batch)  # 分离数据和标签
    lengths = [len(xi) for xi in x]  # 获取每个样本的序列长度
    max_l = max(lengths)  # 批次中的最大序列长度
    # 初始化填充张量，数据用0填充，标签用-1填充以标记无效位置
    padded_x = torch.zeros(len(batch), max_l, x[0].shape[1])
    padded_y = torch.ones(len(batch), max_l, 1) * -1  # -1表示填充位置
    # 将每个样本填充到最大长度
    for i in range(len(batch)):
        padded_x[i, :lengths[i]] = x[i]
        padded_y[i, :lengths[i]] = y[i]
    return padded_x, padded_y




def collate_fn(batch):
    # Pad sequences to the longest in the batch
    # Input: batch (list of (sequence, label, seq_length))
    # Output: padded_data [B, max_len, D], labels [B], seq_lengths (list)
    data, labels, seq_lengths = zip(*batch)
    max_len = max(seq_lengths)
    padded_data = torch.zeros(len(data), max_len, data[0].size(-1))
    for i, seq in enumerate(data):
        padded_data[i, :len(seq)] = seq
    return padded_data, torch.tensor(labels), seq_lengths

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
    def __init__(self, num_samples, d_model, min_len=5, max_len=15):
        # Initialize dataset with variable-length sequences
        self.data = []
        self.labels = []
        self.seq_lengths = []
        for _ in range(num_samples):
            # Random sequence length between min_len and max_len
            seq_len = np.random.randint(min_len, max_len + 1)
            seq = torch.randn(seq_len, d_model)
            # Binary label based on sum parity
            label = (seq.sum(dim=0).sum() % 2).long()
            self.data.append(seq)
            self.labels.append(label)
            self.seq_lengths.append(seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return sequence, label, and sequence length
        return self.data[idx], self.labels[idx], self.seq_lengths[idx]

def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    data = {}
    # 加载数据到data字典中
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'), allow_pickle=True)
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        data['l_' + category] = cat_data['l']
        del cat_data

    train_dataset = VariableLengthDataset(data['x_train'], data['y_train'], data['l_train'])
    val_dataset = VariableLengthDataset(data['x_val'], data['y_val'], data['l_val'])
    test_dataset = VariableLengthDataset(data['x_test'], data['y_test'], data['l_test'])

    scaler = StandardScaler(mean=data['y_train'][..., 0].mean(), std=data['y_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])

    # Generate synthetic variable-length dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, scaler


cat_data = np.load('train' + '.npz', allow_pickle=True)
for key in cat_data.keys():
    print(key)
print(cat_data['x'][:,-1].mean())