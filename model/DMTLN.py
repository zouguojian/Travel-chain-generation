import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from data.Dataload import load_segment_index

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Initialize positional encoding matrix with shape [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # Create position indices: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute scaling factors for sin/cos functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input tensor x: [B, L, D]
        # Only take positions up to sequence length
        return x + self.pe[:, :x.size(1)]

def generate_square_subsequent_mask(sz, device):
    # Generate causal (subsequent) mask for self-attention to prevent attending to future tokens
    # Input: sz (sequence length), device
    # Output: mask [sz, sz] where upper triangle is -inf, lower triangle and diagonal are 0
    '''
    [[0., -inf, -inf, -inf],
     [0., 0., -inf, -inf],
     [0., 0., 0., -inf],
     [0., 0., 0., 0.]]
    '''
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)

def generate_padding_mask(seq_lengths, max_len, device):
    # Generate padding mask to ignore padded positions in attention
    # Input: seq_lengths (list of sequence lengths), max_len (maximum sequence length), device
    # Output: mask [B, max_len], True for padding positions, False for valid positions
    B = len(seq_lengths)
    mask = torch.ones(B, max_len, dtype=torch.bool, device=device)
    for i, length in enumerate(seq_lengths):
        mask[i, :length] = False
    return mask

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model  # Input feature dimension
        self.nhead = nhead      # Number of attention heads
        self.head_dim = d_model // nhead  # Dimension per head
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        # Use Conv1d (kernel_size=1) instead of Linear for Q, K, V projections
        self.q_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.k_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.v_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.out_conv = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, query, key, value, attn_mask=None, padding_mask=None):
        # Input: query, key, value [B, L, D], attn_mask [L, L], padding_mask [B, L]
        # Output: [B, L, D]
        B, L, _ = query.size()
        # Transpose for Conv1d: [B, L, D] -> [B, D, L]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # Apply convolutions and reshape for multi-head: [B, D, L] -> [B, L, nhead, head_dim]
        Q = self.q_conv(query).transpose(1, 2).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_conv(key).transpose(1, 2).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_conv(value).transpose(1, 2).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        # Compute attention scores: [B, nhead, L, L]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Apply causal mask
        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
        # Apply padding mask
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        # Compute attention weights
        attn = torch.softmax(scores, dim=-1)
        # Apply attention to values: [B, nhead, L, head_dim]
        out = torch.matmul(attn, V)
        # Reshape and project output: [B, L, D]
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.out_conv(out.transpose(1, 2)).transpose(1, 2)
        return out

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        # Feed-forward network using Conv1d
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, dim_feedforward, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim_feedforward, d_model, kernel_size=1)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, padding_mask=None):
        # Input: x [B, L, D], attn_mask [L, L], padding_mask [B, L]
        # Output: [B, L, D]
        # Self-attention with residual connection and normalization
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, attn_mask, padding_mask)))
        # Feed-forward with residual connection
        x_t = x.transpose(1, 2)  # [B, D, L] for Conv1d
        x_t = self.ffn(x_t)
        x = self.norm2(x + self.dropout(x_t.transpose(1, 2)))
        return x

class HolisticAttention(nn.Module):
    def __init__(self, d_model, num_layers=6, nhead=8, dim_feedforward=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        # Learnable BOS and EOS tokens
        # self.bos = nn.Parameter(torch.randn(1, 1, d_model))
        # self.eos = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, max_len)
        # Stack multiple transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seq_lengths):
        # Input: x [B, L, D], seq_lengths (list of sequence lengths)
        # Output: [B, L+2, D]
        B, L, _ = x.shape
        # Learnable BOS and EOS tokens
        bos = torch.randn(B, 1, self.d_model, device=x.device)
        eos = torch.randn(B, 1, self.d_model, device=x.device)
        x = torch.cat([bos, x, eos], dim=1)  # [B, L+2, D]
        # Update sequence lengths for BOS and EOS
        seq_lengths = [l + 2 for l in seq_lengths]
        # Apply positional encoding
        x = self.pos_enc(x) * math.sqrt(self.d_model)
        # Generate causal and padding masks
        max_len = x.size(1)
        attn_mask = generate_square_subsequent_mask(max_len, x.device)
        padding_mask = generate_padding_mask(seq_lengths, max_len, x.device)
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask, padding_mask)
        # Final normalization
        x = self.norm(x)
        return x # [B, L+2, D]

class SpatiotemporalModel(nn.Module):
    def __init__(self,):
        super(SpatiotemporalModel, self).__init__()

    def forward(self, x):
        return x

class MultiTaskModel(nn.Module):
    def __init__(self, in_features, hidden_channels=64):
        super(MultiTaskModel, self).__init__()

        # Task 1: Regression [B, L]
        self.task1 = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_channels, out_channels=1, kernel_size=3, padding=1)
        )

        # Task 2: Regression [B, 1]
        self.task2 = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_channels, out_channels=1, kernel_size=3, padding=1),
        )

        # Task 3: Classification [B, L] with Softmax on L dimension
        self.task3 = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_channels, out_channels=1, kernel_size=3, padding=1)
        )
        # self.softmax = nn.Softmax(dim=1)  # Softmax along L dimension

    def forward(self, x1, x2, x3 = None):
        # x1: [B, L, D], x2: [B, D, 1]

        # Task 1
        x_perm = x1.permute(0, 2, 1)  # [B, D, L]
        task1_out = self.task1(x_perm).squeeze(1)  # [B, 1, L] -> [B, L]

        # Task 2
        task2_out = self.task2(x2).squeeze(1)  # [B, 1]

        # Task 3
        t3 = self.task3(x_perm)  # [B, 1, L]
        task3_out = t3.squeeze(1)  # [B, L]
        # task3_out = self.softmax(task3_out)  # [B, L], probabilities sum to 1 along L

        return task1_out, task2_out, task3_out

# DeepFIN模型定义
class DeepFinModel(nn.Module):
    def __init__(self, field_dims, num_features, embed_dim=32, hidden_dims=[64, 32], dropout=0.2):
        super(DeepFinModel, self).__init__()
        self.field_dims = field_dims  # 类别特征的field_dims
        self.num_cat = len(field_dims) # 类别特征占用的长度
        self.num_cont = num_features - self.num_cat # 浮点型特征占用的长度
        self.embed_dim = embed_dim  # 嵌入的维度
        self.total_fields = num_features + 1 # 总特征长度, 额外的"1"是交通状态

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

    def forward(self, x, states):
        # x shape: [B, Max_L, num_features]
        # states shape: [B, Max_L, embed_dim]
        # 重塑为 [B, num_features, Max_L] 以适配Conv1d
        x = x.permute(0, 2, 1)  # [B, num_features, Max_L]
        states = states.permute(0, 2, 1)  # [B, embed_dim, Max_L], 后续加入到路段建模中

        # 类别特征嵌入
        cat_emb_list = [self.cat_embeds[i](x[:, i, :].long()).permute(0, 2, 1)  # [B, embed_dim, Max_L]
                        for i in range(self.num_cat)]

        # 连续特征嵌入
        cont_emb_list = [self.cont_embeds[j](x[:, self.num_cat + j:self.num_cat + j + 1, :])  # [B, embed_dim, Max_L]
                         for j in range(self.num_cont)]

        # 所有嵌入 for FIN interaction
        all_emb_list = cat_emb_list + cont_emb_list + [states]  # list of [B, embed_dim, Max_L]
        embed_stacked = torch.stack(all_emb_list, dim=2)  # [B, embed_dim, total_fields, Max_L]

        # FIN 二阶交互
        square_of_sum = torch.sum(embed_stacked, dim=2) ** 2  # [B, embed_dim, Max_L + 1]
        sum_of_square = torch.sum(embed_stacked ** 2, dim=2)  # [B, embed_dim, Max_L + 1]
        fin_interaction = 0.5 * (square_of_sum - sum_of_square)  # [B, embed_dim, Max_L + 1]

        # FIN 线性部分
        cat_linear_terms = [self.cat_linears[i](x[:, i, :].long()).permute(0, 2, 1) for i in
                            range(self.num_cat)]  # list [B, embed_dim, Max_L]
        cont_linear_terms = [self.cont_linears[j](x[:, self.num_cat + j:self.num_cat + j + 1, :])  # [B, embed_dim, Max_L]
                             for j in range(self.num_cont)]

        linear_terms = sum(cat_linear_terms + cont_linear_terms + [states]) + self.fin_bias.view(1, -1, 1)  # [B, embed_dim, Max_L]

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

# Multi-Layer Multi-Head Attention for Spatial
class CustomMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'wq': nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
                'wk': nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
                'wv': nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
                'wo': nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
                'norm': nn.LayerNorm(embed_dim)
            }) for _ in range(num_layers)
        ])

    def forward(self, x, adj=None):
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {embed_dim}"

        for layer in self.layers:
            x_in = x.unsqueeze(-1).permute(0, 2, 1, 3)  # [batch_size, embed_dim, seq_len, 1]

            Q = layer['wq'](x_in).permute(0, 2, 3, 1).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
            K = layer['wk'](x_in).permute(0, 2, 3, 1).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            V = layer['wv'](x_in).permute(0, 2, 3, 1).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, seq_len, seq_len]

            if adj is not None:
                assert adj.shape == (seq_len, seq_len), f"Expected adj shape ({seq_len}, {seq_len}), got {adj.shape}"
                scores = scores * adj.unsqueeze(0).unsqueeze(0)

            attn = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
            out = torch.matmul(attn, V)  # [batch_size, num_heads, seq_len, head_dim]
            out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, embed_dim)  # [batch_size, seq_len, embed_dim]

            out = out.unsqueeze(-1).permute(0, 2, 1, 3)  # [batch_size, embed_dim, seq_len, 1]
            out = layer['wo'](out).permute(0, 2, 3, 1).squeeze(-2)  # [batch_size, seq_len, embed_dim]

            x = layer['norm'](x + out)  # [batch_size, seq_len, embed_dim]

        return x

# Multi-Layer Temporal Multi-Head Attention
class CustomTemporalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'wq': nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
                'wk': nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
                'wv': nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
                'wo': nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
                'norm': nn.LayerNorm(embed_dim)
            }) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {embed_dim}"

        for layer in self.layers:
            x_in = x  # [batch_size, seq_len, embed_dim]

            # Reshape for Conv2d: [batch_size, embed_dim, seq_len, 1]
            x_conv = x_in.unsqueeze(-1).permute(0, 2, 1, 3)  # [batch_size, embed_dim, seq_len, 1]

            Q = layer['wq'](x_conv).permute(0, 2, 3, 1).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
            K = layer['wk'](x_conv).permute(0, 2, 3, 1).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            V = layer['wv'](x_conv).permute(0, 2, 3, 1).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, seq_len, seq_len]

            if mask is not None:
                assert mask.shape == (seq_len, seq_len), f"Expected mask shape ({seq_len}, {seq_len}), got {mask.shape}"
                extended_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
                scores = scores.masked_fill(extended_mask, float('-inf'))

            attn = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
            out = torch.matmul(attn, V)  # [batch_size, num_heads, seq_len, head_dim]
            out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, embed_dim)  # [batch_size, seq_len, embed_dim]

            # Apply output convolution separately
            out_conv = out.unsqueeze(-1).permute(0, 2, 1, 3)  # [batch_size, embed_dim, seq_len, 1]
            out = layer['wo'](out_conv).permute(0, 2, 3, 1).squeeze(-2)  # [batch_size, seq_len, embed_dim]

            x = layer['norm'](x_in + out)  # [batch_size, seq_len, embed_dim]

        return x

# Multi-Layer TCN with exponentially increasing dilation
class SimpleTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                dilation=(2 ** i, 1),
                bias=True
            ) for i in range(num_layers)
        ])

        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            dilation = 2 ** i
            padding = ((self.kernel_size - 1) * dilation, 0)
            x = F.pad(x, (0, 0, padding[0], 0))
            x = conv(x)
            x = F.relu(x)

        x = self.final_conv(x)
        x = x[:, :, -1:, :]  # [B, D, 1, N]
        return x

class STEmbedding(nn.Module):
    '''
    spatio-temporal embedding
    TE:     [batch_size, num_his, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, num_his, num_vertex, D]
    '''

    def __init__(self, D_in, D_out, N_1, N_2 = 7, N_3 = 288):
        # N_1表示路段数, N_2表示监测站点数量, N_2表示一周多少天, N_3表示一天多少分钟数
        super(STEmbedding, self).__init__()
        self.SE_emd = nn.Embedding(N_1, D_out)
        self.DOW_emd = nn.Embedding(N_2, D_out)
        self.MOD_emd = nn.Embedding(N_3, D_out)
        self.FC = nn.Sequential(
            nn.Conv2d(D_in, D_out, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(D_out, D_out, kernel_size=1, bias=True),
        )

    def forward(self, SE, TE, T=288):
        # SE输入, 时间dow和mod输入分别为 [B, T, 1]
        # TE输入为[1, N]
        # 输出为 [B, T, N, D]
        # spatial embedding
        SE = self.SE_emd(SE).unsqueeze(0).unsqueeze(0) # [1, N]
        # temporal embedding
        DOWE = self.DOW_emd(TE[:, 0]).unsqueeze(0)
        MODE = self.MOD_emd(TE[:, 1]).unsqueeze(0)
        TE = torch.cat((DOWE, MODE), dim=-1)
        TE = self.FC(TE)
        return SE + TE

# ST-Block Module with initial embedding and output projection
class STBlock(nn.Module):
    def __init__(self, D_in=1, D_out=32, num_heads=8):
        super().__init__()
        self.D = D_out
        self.num_heads = num_heads

        # Initial embedding convolutions: D_in=1 to D_out=32
        self.init_emb = nn.Sequential(
            nn.Conv2d(D_in, 16, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(16, D_out, kernel_size=1, bias=True)
        )

        # Temporal Attention
        self.temporal_attn = CustomTemporalMultiHeadAttention(embed_dim=D_out, num_heads=num_heads, num_layers=2)

        # Spatial Attention
        self.spatial_attn = CustomMultiHeadAttention(embed_dim=D_out, num_heads=num_heads, num_layers=2)

        # Fusion CNN
        self.fusion_cnn = nn.Sequential(
            nn.Conv2d(2 * D_out, D_out, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(D_out, D_out, kernel_size=1)
        )

        # Gate with 1x1 Conv
        self.gate_conv1 = nn.Conv2d(2 * D_out, D_out, kernel_size=1)
        self.gate_conv2 = nn.Conv2d(D_out, D_out, kernel_size=1)

        # TCN
        self.tcn = SimpleTCN(D_out, D_out, kernel_size=3, num_layers=3)

        # Output projection: D_out=32 to D_in=1
        self.output_conv = nn.Conv2d(D_out, D_out, kernel_size=1, bias=True)

    def forward(self, x, adj1, adj2):
        # X输入为 [B, T, N, 1]
        # 时间dow和mod输入 [B, T, 1]
        B, T, N, _ = x.shape

        # Apply embedding convolutions
        x = x.permute(0, 3, 1, 2)  # [B, D_in, T, N]
        x = self.init_emb(x)       # [B, D_out, T, N]
        x = x.permute(0, 2, 3, 1)  # [B, T, N, D_out]

        # Temporal Attention
        x_temp = x.permute(0, 2, 1, 3).reshape(B * N, T, self.D)  # [B*N, T, D_out]
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        hdt = self.temporal_attn(x_temp, mask=causal_mask)
        assert hdt.shape == (B * N, T, self.D), f"Expected hdt shape ({B * N}, {T}, {self.D}), got {hdt.shape}"
        hdt = hdt.view(B, N, T, self.D).permute(0, 2, 1, 3)  # [B, T, N, D_out]

        # Spatial Attention
        x_spat = x.permute(0, 1, 2, 3).reshape(B * T, N, self.D)  # [B*T, N, D_out]
        hdg = self.spatial_attn(x_spat, adj=adj1)
        hda = self.spatial_attn(x_spat, adj=adj2)
        assert hdg.shape == (B * T, N, self.D), f"Expected hdg shape ({B * T}, {N}, {self.D}), got {hdg.shape}"
        assert hda.shape == (B * T, N, self.D), f"Expected hda shape ({B * T}, {N}, {self.D}), got {hda.shape}"
        hdg = hdg.view(B, T, N, self.D)  # [B, T, N, D_out]
        hda = hda.view(B, T, N, self.D)  # [B, T, N, D_out]

        # Multi-Feature Fusion
        h_concat = torch.cat([hdg, hda], dim=-1)  # [B, T, N, 2*D_out]
        h_concat = h_concat.permute(0, 3, 1, 2)  # [B, 2*D_out, T, N]
        hds = self.fusion_cnn(h_concat)  # [B, D_out, T, N]
        hds = hds.permute(0, 2, 3, 1)  # [B, T, N, D_out]

        # Gate mechanism
        gate_input = torch.cat([hdt, hds], dim=-1)  # [B, T, N, 2*D_out]
        gate_input = gate_input.permute(0, 3, 1, 2)  # [B, 2*D_out, T, N]
        z = torch.sigmoid(self.gate_conv1(gate_input))  # [B, D_out, T, N]
        z = self.gate_conv2(z)  # [B, D_out, T, N]
        z = z.permute(0, 2, 3, 1)  # [B, T, N, D_out]
        hst = z * hdt + (1 - z) * hds  # [B, T, N, D_out]

        # TCN
        hst_tcn = hst.permute(0, 3, 1, 2)  # [B, D_out, T, N]
        hsf = self.tcn(hst_tcn)  # [B, D_out, 1, N]
        hsf = self.output_conv(hsf)  # [B, D_in, 1, N]
        hsf = hsf.permute(0, 2, 3, 1)  # [B, 1, N, D_in]
        # hsf = hsf.squeeze(1)  # [B, N, D_in]
        return hsf


# Define the fusion module
class FlowSpeedFusion(nn.Module):
    def __init__(self, D, hidden_dim = 32):
        super(FlowSpeedFusion, self).__init__()
        self.conv1 = nn.Conv2d(3 * D, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, XS, XF, in_stations, out_stations):
        # XS: [B, T, 108, D]
        # XF: [B, T, 66, D]
        # Gather start and end flows
        start_flow = XF[:, :, in_stations]  # [B, T, 108, D]
        end_flow = XF[:, :, out_stations]  # [B, T, 108, D]

        # Concatenate: [start_flow, XS, end_flow]
        XSF = torch.cat([start_flow, XS, end_flow], dim=-1)  # [B, T, 108, 3*D]

        # Apply two convolution layers (pointwise)
        B, T, N, F = XSF.shape
        x = XSF.permute(0, 3, 1, 2)  # [B, 3*D, T, N]
        x = self.conv1(x)  # [B, hidden_dim, T, N]
        x = self.relu(x)
        x = self.conv2(x)  # [B, hidden_dim, T, N]
        X = x.permute(0, 2, 3, 1)  # [B, T, N, hidden_dim]

        return X

def segment_states_paded(x, indices, max_l):
    '''
     x shape: (B, T, N, D)
     indices: [array([80], dtype=int32), array([28, 30], dtype=int32), ...]
     return shape: (B, R, L, D)
    '''
    B, T, N, D = x.shape
    padded_x = torch.zeros(B, indices.shape[0], max_l, D, device=x.device)  # [B, R, max_L, D]
    for i in range(indices.shape[0]):
        padded_x[:, i, :indices[i].shape[0]] = x[:, -1, indices[i]]

    return padded_x # torch.Size([32, 10, 8, 32]), shape: [B, R, L, D]

class DmtlnModel(nn.Module):
    def __init__(self, field_dims,
                 num_features,
                 embed_size = 32,
                 class_num = 10,
                 N_spped = 106,
                 N_flow = 66,
                 max_len = 15, ytra_mean = 0.0, ytra_std = 1.0, ytol_mean =0.0, ytol_std = 1.0):
        super(DmtlnModel, self).__init__()
        self.ytra_mean = ytra_mean
        self.ytra_std = ytra_std
        self.ytol_mean = ytol_mean
        self.ytol_std = ytol_std
        self.embed_size = embed_size

        # 交通状态网络定义, 包括对路段的速度建模和对站点交通速度建模
        self.statemodel = STBlock(D_in = 1, D_out = embed_size, num_heads = 4)
        # 交通状态融合模块, 速度与流量融合
        self.flowspeedfusion = FlowSpeedFusion(D = embed_size, hidden_dim = embed_size)

        # 特征交叉网络定义
        self.finmodel = DeepFinModel(field_dims, num_features, embed_dim=embed_size, hidden_dims=[64, embed_size], dropout=0.2)

        # 全局注意力定义，这里不使用多头注意力，但是使用了mask和position机制
        self.holisticatt = HolisticAttention(embed_size, 1, 1, embed_size * 2, 0.2, max_len)

        # 多任务模块定义
        self.multitask = MultiTaskModel(in_features = embed_size, hidden_channels=64)

    def forward(self, x, x_flow, x_speed, batch_dow, batch_mod, batch_cla, seq_lengths, df, adjs = []):
        # x: [batch_size, routes, max_length, num_fields]
        # Generate output
        # print('x shape', x.shape, 'batch_flow shape', batch_flow.shape, 'batch_speed shape', batch_speed.shape, 'batch_dow shape', batch_dow.shape, 'batch_mod shape', batch_mod.shape, 'batch_cla shape', batch_cla.shape)
        batch_size, routes, max_length, num_fields = x.shape
        copy = x
        # 重塑为 [batch_size * routes, max_length, num_fields]
        x = x.view(batch_size * routes, max_length, num_fields)

        x_flow, x_speed = x_flow.unsqueeze(3), x_speed.unsqueeze(3) # [B, T, N, 1]
        xf = self.statemodel(x_flow, adj1 = adjs[0], adj2 = adjs[1])   # [B, 1, 66, D]
        xs = self.statemodel(x_speed, adj1 = adjs[2], adj2 = adjs[3])  # [B, 1, 108, D]
        # Extract in_station and out_station as tensors
        in_stations = torch.tensor(df['in_station'].values, dtype=torch.int32, device=x.device)
        out_stations = torch.tensor(df['out_station'].values, dtype=torch.int32, device=x.device)
        xst = self.flowspeedfusion(XS = xs, XF = xf, in_stations = in_stations, out_stations = out_stations) # [B, 1, 108, D]

        segment_indices = load_segment_index()
        segment_states = segment_states_paded(xst, segment_indices, max_length)  # [B, R, max_length, D] [32, 10, 8, 32]
        segment_states = segment_states.view(batch_size * routes, max_length, self.embed_size)  # [batch_size * routes, max_length, D]
        # 经过FIN进行路段级别特征提取
        x = self.finmodel(x, segment_states) # [batch_size * routes, max_length, num_fields]
        _, new_length, new_dim = x.shape
        x = x.view(batch_size, routes, new_length, new_dim)[torch.arange(batch_size), batch_cla.view(batch_size)]
        x = self.holisticatt(x, seq_lengths)  # [batch_size, L+2, D], [B]

        # Select all valid token representation for travel time on each segment
        max_len = max(seq_lengths)
        valid_rep = x[torch.arange(x.size(0)), 1 : max_len + 1] # [B, L, D]

        # Select EOS token representation for total travel time
        eos_indices = torch.tensor([l + 1 for l in seq_lengths], device=x.device)
        eos_rep = x[torch.arange(x.size(0)), eos_indices].unsqueeze(-1)  # [B, D]

        results = self.multitask(valid_rep, eos_rep)

        return (results[0] * self.ytra_std) + self.ytra_mean, (results[1] * self.ytol_std) + self.ytol_mean, results[2]