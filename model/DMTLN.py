import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatiotemporalModel(nn.Module):
    def __init__(self,):
        super(SpatiotemporalModel, self).__init__()

    def forward(self, x):
        return x

class MultiTaskModel(nn.Module):
    def __init__(self, in_features, hidden_channels=64):
        super(MultiTaskModel, self).__init__()

        # Task 1: Classification [B, L] with Softmax on L dimension
        self.task1 = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_channels, out_channels=1, kernel_size=3, padding=1)
        )
        self.softmax = nn.Softmax(dim=1)  # Softmax along L dimension

        # Task 2: Regression [B, L, 1]
        self.task2 = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_channels, out_channels=1, kernel_size=3, padding=1)
        )

        # Task 3: Regression [B, 1, 1]
        self.task3 = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_channels, out_channels=1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: [B, L, D]
        x_perm = x.permute(0, 2, 1)  # [B, D, L]
        # Task 1
        t1 = self.task1(x_perm)  # [B, 1, L]
        task1_out = t1.squeeze(1)  # [B, L]
        task1_out = self.softmax(task1_out)  # [B, L], probabilities sum to 1 along L

        # Task 2
        task2_out = self.task2(x_perm).permute(0, 2, 1)  # [B, 1, L] -> [B, L, 1]

        # Task 3
        task3_out = self.task3(x_perm)  # [B, 1, 1]

        return task1_out, task2_out, task3_out

# DeepFIN模型定义
class DeepFinModel(nn.Module):
    def __init__(self, field_dims, num_features, embed_dim=32, hidden_dims=[64, 32], dropout=0.2):
        super(DeepFinModel, self).__init__()
        self.field_dims = field_dims  # 类别特征的field_dims
        self.num_cat = len(field_dims)
        self.num_cont = num_features - self.num_cat
        self.embed_dim = embed_dim
        self.total_fields = self.num_cat + self.num_cont

        # 嵌入层 for FM and DNN
        self.cat_embeds = nn.ModuleList([nn.Embedding(fd, embed_dim) for fd in field_dims])
        self.cont_embeds = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=1) for _ in range(self.num_cont)
        ])

        # Linear for FM
        self.cat_linears = nn.ModuleList([nn.Embedding(fd, embed_dim) for fd in field_dims])
        self.cont_linears = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=1) for _ in range(self.num_cont)
        ])
        self.fm_bias = nn.Parameter(torch.zeros(embed_dim))

        # DNN部分
        total_embed_dim = self.total_fields * embed_dim  # Use flattened embeds for all fields
        self.dnn_layers = nn.ModuleList()
        prev_dim = total_embed_dim
        for hidden_dim in hidden_dims:
            self.dnn_layers.append(nn.Sequential(
                nn.Conv1d(in_channels=prev_dim, out_channels=hidden_dim, kernel_size=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            prev_dim = hidden_dim
        self.dnn_output = nn.Conv1d(in_channels=prev_dim, out_channels=embed_dim, kernel_size=1)

    def forward(self, x):
        # x shape: [B, num_features]
        # output shape: [B, embed_dim]
        # 重塑输入为 [B, num_features, 1]
        x = x.unsqueeze(-1)  # [B, num_features, 1]

        # 类别特征嵌入
        cat_emb_list = [self.cat_embeds[i](x[:, i, 0].long()) for i in range(self.num_cat)]  # list [B, embed_dim]
        # 连续特征"嵌入" (Conv1d)
        cont_emb_list = [self.cont_embeds[j](x[:, self.num_cat + j:self.num_cat + j + 1])  # [B, embed_dim, 1]
                         for j in range(self.num_cont)]
        cont_emb_list = [emb.squeeze(-1) for emb in cont_emb_list]  # list [B, embed_dim]

        # 所有嵌入 for FM interaction
        all_emb_list = cat_emb_list + cont_emb_list
        embed_stacked = torch.stack(all_emb_list, dim=1)  # [B, total_fields, embed_dim]

        # FM 二阶交互
        square_of_sum = torch.sum(embed_stacked, dim=1) ** 2  # [B, embed_dim]
        sum_of_square = torch.sum(embed_stacked ** 2, dim=1)  # [B, embed_dim]
        fm_interaction = 0.5 * (square_of_sum - sum_of_square)  # [B, 1]

        # FM 线性部分
        cat_linear_terms = [self.cat_linears[i](x[:, i, 0].long()) for i in range(self.num_cat)]  # list [B, 1]
        cont_linear_terms = [self.cont_linears[j](x[:, self.num_cat + j:self.num_cat + j + 1])  # [B, 1, 1]
                             for j in range(self.num_cont)]
        cont_linear_terms = [term.squeeze(-1) for term in cont_linear_terms]  # list [B, 1]
        linear_terms = sum(cat_linear_terms + cont_linear_terms) + self.fm_bias  # [B, 1]

        # DNN 部分: 使用所有字段的嵌入
        dnn_input = torch.cat(all_emb_list, dim=1).unsqueeze(-1)  # [B, total_fields * embed_dim, 1]
        dnn_out = dnn_input
        for layer in self.dnn_layers:
            dnn_out = layer(dnn_out)
        dnn_out = self.dnn_output(dnn_out).squeeze(-1)  # [B, 1]

        # 合并输出
        print(linear_terms.shape, fm_interaction.shape, dnn_out.shape)
        output = linear_terms + fm_interaction + dnn_out
        return output

class DmtlnModel(nn.Module):
    def __init__(self, field_dims, num_features, embedding_size = 32, hidden_size=[64, 32], class_num = 11):
        super(DmtlnModel, self).__init__()
        self.task_1 = nn.Linear(hidden_size, class_num) # 出行链选择
        self.finmodel = DeepFinModel(field_dims, num_features, embed_dim=32, hidden_dims=[64, 32], dropout=0.2)

    def forward(self, x):
        # x: [batch_size, routes, max_length, num_fields]
        # Generate output
        x = F.softmax(self.finmodel(x)) # x:[-1, ]
        return x