import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


# 自定义数据集类
class DeepFMDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# DeepFM模型定义
class DeepFM(nn.Module):
    def __init__(self, field_dims, num_features, embed_dim=32, hidden_dims=[64, 32], dropout=0.2):
        super(DeepFM, self).__init__()
        self.field_dims = field_dims  # 类别特征的field_dims
        num_cat = len(field_dims)
        num_cont = num_features - num_cat
        self.num_cat = num_cat
        self.num_cont = num_cont
        self.embed_dim = embed_dim
        total_fields = num_cat + num_cont

        # 嵌入层 for FM and DNN
        self.cat_embeds = nn.ModuleList([nn.Embedding(fd, embed_dim) for fd in field_dims])
        self.cont_embeds = nn.ModuleList([nn.Linear(1, embed_dim, bias=False) for _ in
                                          range(num_cont)])  # Equivalent to pointwise convolution for unity

        # Linear for FM
        self.cat_linears = nn.ModuleList([nn.Embedding(fd, 1) for fd in field_dims])
        self.cont_linears = nn.ModuleList([nn.Linear(1, 1) for _ in range(num_cont)])
        self.fm_bias = nn.Parameter(torch.zeros(1))

        # DNN部分
        total_embed_dim = total_fields * embed_dim  # Use flattened embeds for all fields
        self.dnn_layers = nn.ModuleList()
        prev_dim = total_embed_dim
        for hidden_dim in hidden_dims:
            self.dnn_layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            prev_dim = hidden_dim
        self.dnn_output = nn.Linear(prev_dim, 1)

    def forward(self, x):
        # x shape: [B, num_features]
        # 类别特征嵌入
        cat_emb_list = [self.cat_embeds[i](x[:, i].long()) for i in range(self.num_cat)]  # list [B, embed_dim]
        # 连续特征"嵌入" (v * x)
        cont_emb_list = [self.cont_embeds[j](x[:, self.num_cat + j].unsqueeze(1)) for j in
                         range(self.num_cont)]  # list [B, embed_dim]

        # 所有 weighted embeds for FM interaction
        all_emb_list = cat_emb_list + cont_emb_list
        embed_stacked = torch.stack(all_emb_list, dim=1)  # [B, total_fields, embed_dim]

        # FM 二阶交互
        square_of_sum = torch.sum(embed_stacked, dim=1) ** 2  # [B, embed_dim]
        sum_of_square = torch.sum(embed_stacked ** 2, dim=1)  # [B, embed_dim]
        fm_interaction = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)  # [B, 1]

        # FM 线性部分
        cat_linear_terms = [self.cat_linears[i](x[:, i].long()) for i in range(self.num_cat)]
        cont_linear_terms = [self.cont_linears[j](x[:, self.num_cat + j].unsqueeze(1)) for j in range(self.num_cont)]
        linear_terms = sum(cat_linear_terms + cont_linear_terms) + self.fm_bias  # [B, 1]

        # DNN 部分: 使用所有字段的嵌入 (flattened)
        dnn_input = torch.cat(all_emb_list, dim=1)  # [B, total_fields * embed_dim]
        dnn_out = dnn_input
        for layer in self.dnn_layers:
            dnn_out = layer(dnn_out)
        dnn_out = self.dnn_output(dnn_out)  # [B, 1]

        # 合并输出
        output = linear_terms + fm_interaction + dnn_out
        return torch.sigmoid(output)


# 训练函数
def train_deepfm(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu', save_path='deepfm_model.pth'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x).squeeze()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x).squeeze()
                loss = criterion(output, batch_y)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f'Saved best model at epoch {epoch + 1}')


# 示例使用
if __name__ == "__main__":
    # 超参数
    field_dims = [300, 1009, 45, 98]  # 前四个field的取值范围
    num_features = 10  # 总特征维度 D（4个整型 + 6个浮点型）
    embed_dim = 32
    hidden_dims = [64, 32]
    dropout = 0.2
    batch_size = 32
    epochs = 10
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模拟数据
    np.random.seed(42)
    num_samples = 1000
    data = np.zeros((num_samples, num_features))
    data[:, 0] = np.random.randint(0, 300, size=(num_samples,))
    data[:, 1] = np.random.randint(0, 1009, size=(num_samples,))
    data[:, 2] = np.random.randint(0, 45, size=(num_samples,))
    data[:, 3] = np.random.randint(0, 98, size=(num_samples,))
    data[:, 4:] = np.random.randn(num_samples, num_features - 4)  # 浮点型特征
    labels = np.random.randint(0, 2, size=(num_samples,))  # 二分类标签

    # 创建数据集
    dataset = DeepFMDataset(data, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 初始化模型
    model = DeepFM(
        field_dims=field_dims,
        num_features=num_features,
        embed_dim=embed_dim,
        hidden_dims=hidden_dims,
        dropout=dropout
    )

    # 训练模型
    train_deepfm(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)