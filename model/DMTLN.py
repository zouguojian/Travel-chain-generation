import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None):
        B, L, _ = query.size()
        Q = self.q_linear(query).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.out_linear(out)
        return out

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, attn_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class HolisticAttention(nn.Module):
    def __init__(self, d_model, num_layers=6, nhead=8, dim_feedforward=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.bos = nn.Parameter(torch.randn(1, 1, d_model))
        self.eos = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):

        B, L, _ = x.shape
        # Add BOS and EOS tokens
        bos = self.bos.repeat(B, 1, 1)
        eos = self.eos.repeat(B, 1, 1)
        x = torch.cat([bos, x, eos], dim=1)  # [B, L+2, D]

        # Apply positional encoding
        x = self.pos_enc(x) * math.sqrt(self.d_model)

        # Generate causal mask
        mask = generate_square_subsequent_mask(x.size(1)).to(x.device)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final normalization
        x = self.norm(x)
        return x



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
        self.num_cat = len(field_dims) # 类别特征占用的长度
        self.num_cont = num_features - self.num_cat # 浮点型特征占用的长度
        self.embed_dim = embed_dim
        self.total_fields = self.num_cat + self.num_cont

        # 嵌入层 for FIN and DNN
        self.cat_embeds = nn.ModuleList([nn.Embedding(fd, embed_dim) for fd in field_dims])
        self.cont_embeds = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=1) for _ in range(self.num_cont)
        ])

        # Linear for FIN
        self.cat_linears = nn.ModuleList([nn.Embedding(fd, embed_dim) for fd in field_dims])
        self.cont_linears = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=1) for _ in range(self.num_cont)
        ])
        self.fin_bias = nn.Parameter(torch.zeros(embed_dim))

        # DNN部分
        total_embed_dim = self.total_fields * embed_dim
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
        # x shape: [B, Max_L, num_features]
        # 重塑为 [B, num_features, Max_L] 以适配Conv1d
        x = x.permute(0, 2, 1)  # [B, num_features, Max_L]

        # 类别特征嵌入
        cat_emb_list = [self.cat_embeds[i](x[:, i, :].long()).permute(0, 2, 1)  # [B, embed_dim, Max_L]
                        for i in range(self.num_cat)]

        # 连续特征嵌入
        cont_emb_list = [self.cont_embeds[j](x[:, self.num_cat + j:self.num_cat + j + 1, :])  # [B, embed_dim, Max_L]
                         for j in range(self.num_cont)]

        # 所有嵌入 for FIN interaction
        all_emb_list = cat_emb_list + cont_emb_list  # list of [B, embed_dim, Max_L]
        embed_stacked = torch.stack(all_emb_list, dim=2)  # [B, embed_dim, total_fields, Max_L]

        # FIN 二阶交互
        square_of_sum = torch.sum(embed_stacked, dim=2) ** 2  # [B, embed_dim, Max_L]
        sum_of_square = torch.sum(embed_stacked ** 2, dim=2)  # [B, embed_dim, Max_L]
        fin_interaction = 0.5 * (square_of_sum - sum_of_square)  # [B, embed_dim, Max_L]

        # FIN 线性部分
        cat_linear_terms = [self.cat_linears[i](x[:, i, :].long()).permute(0, 2, 1) for i in
                            range(self.num_cat)]  # list [B, embed_dim, Max_L]
        cont_linear_terms = [self.cont_linears[j](x[:, self.num_cat + j:self.num_cat + j + 1, :])  # [B, embed_dim, Max_L]
                             for j in range(self.num_cont)]
        print(len(cat_linear_terms), cat_linear_terms[0].shape, len(cont_linear_terms), cont_linear_terms[0].shape)
        linear_terms = sum(cat_linear_terms + cont_linear_terms) + self.fin_bias.view(1, -1, 1)  # [B, embed_dim, Max_L]

        # DNN 部分
        dnn_input = torch.cat(all_emb_list, dim=1)  # [B, total_fields * embed_dim, Max_L]
        dnn_out = dnn_input
        for layer in self.dnn_layers:
            dnn_out = layer(dnn_out)
        dnn_out = self.dnn_output(dnn_out)  # [B, embed_dim, Max_L]

        # 合并输出
        output = linear_terms + fin_interaction + dnn_out  # [B, embed_dim, Max_L]
        results = output.permute(0, 2, 1)
        return results  # [B, Max_L, embed_dim]

class DmtlnModel(nn.Module):
    def __init__(self, field_dims,
                 num_features,
                 embed_size = 32,
                 hidden_size=[64, 32],
                 class_num = 11,
                 max_len = 12):
        super(DmtlnModel, self).__init__()
        self.task_1 = nn.Linear(hidden_size, class_num) # 出行链选择

        # 特征交叉网络定义
        self.finmodel = DeepFinModel(field_dims, num_features, embed_dim=32, hidden_dims=[64, 32], dropout=0.2)

        # 全局注意力定义，这里不使用多头注意力，但是使用了mask和position机制
        self.holisticatt = HolisticAttention(embed_size, 1, 1, embed_size * 2, 0.2, max_len)

        # 多任务模块定义
        self.multitask = MultiTaskModel(in_features = embed_size, hidden_channels=64)

    def forward(self, x):
        # x: [batch_size, routes, max_length, num_fields]
        # Generate output
        batch_size, routes, max_length, num_fields = x.shape
        # 重塑为 [batch_size * routes, max_length, num_fields]
        x = x.view(batch_size * routes, max_length, num_fields)

        # 经过FIN进行路段级别特征提取
        x = self.finmodel(x) # [batch_size * routes, max_length, num_fields]
        return x