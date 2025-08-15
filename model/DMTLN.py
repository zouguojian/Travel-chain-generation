import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DeepFM(nn.Module):
    """
    """

    def __init__(self, feature_fields, embed_dim, mlp_dims, dropout):
        super(DeepFM, self).__init__()
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype=np.long)

        # FM中的线性部分
        self.linear = torch.nn.Embedding(sum(feature_fields) + 1, 1)
        self.bias = torch.nn.Parameter(torch.zeros((1,)))

        # Embedding层
        self.embedding = torch.nn.Embedding(sum(feature_fields) + 1, embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

        # DNN部分
        self.embedding_out_dim = len(feature_fields) * embed_dim
        layers = []
        input_dim = self.embedding_out_dim
        for mlp_dim in mlp_dims:
            # 全连接层
            layers.append(nn.Linear(input_dim, mlp_dim))
            layers.append(nn.BatchNorm1d(mlp_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = mlp_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        """
        tmp = x + x.new_tensor(self.offsets).unsqueeze(0)

        # embedding
        embeddings = self.embedding(tmp)

        # FM
        ## linear part
        linear_part = torch.sum(self.linear(tmp), dim=1) + self.bias
        ## inner part
        square_of_sum = torch.sum(embeddings, dim=1) ** 2
        sum_of_square = torch.sum(embeddings ** 2, dim=1)
        inner_part = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        fm_part = linear_part + inner_part

        # DNN part
        mlp_part = self.mlp(embeddings.view(-1, self.embedding_out_dim))

        # 输出part
        x = fm_part + mlp_part
        x = torch.sigmoid(x.squeeze(1))
        return x

class DMTLN(nn.Module):
    def __init__(self, hidden_size, embedding_size = 64, class_num = 8, ):
        super(DMTLN, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.task_1 = nn.Linear(hidden_size, class_num) # 出行链选择
        model = DeepFM.DeepFM(feature_fields=fields, embed_dim=8, mlp_dims=(32, 16), dropout=0.2)

    def forward(self, x):
        # Generate output
        x = F.softmax(x)
        return x