import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# Multi-layer Multi-head Attention Model for feature extraction and classification
class MultiLayerAttention(nn.Module):
    def __init__(self, field_dims=[], num_features = 7, embed_dim=32, num_classes=10):
        super(MultiLayerAttention, self).__init__()
        self.num_cat = len(field_dims)
        self.num_cont = num_features - self.num_cat
        self.embed_dim = embed_dim

        # Embeddings for categorical features
        self.cat_embeds = nn.ModuleList([nn.Embedding(fd, embed_dim) for fd in field_dims])

        # Conv1d for continuous features
        self.cont_embeds = nn.ModuleList(
            [nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=1) for _ in range(self.num_cont)])

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=128, dropout=0.1,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Classifier
        self.fc1 = nn.Linear(embed_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input: [B, L]
        cat_x = x[:, :self.num_cat].long()  # [B, num_cat]
        cont_x = x[:, self.num_cat:].float()  # [B, num_cont]

        embeds = []
        # Embed categorical features
        for j in range(self.num_cat):
            embed = self.cat_embeds[j](cat_x[:, j])  # [B, embed_dim]
            embeds.append(embed)

        # Embed continuous features
        for j in range(self.num_cont):
            input_cont = cont_x[:, j].unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            embed = self.cont_embeds[j](input_cont)  # [B, embed_dim, 1]
            embed = embed.squeeze(2)  # [B, embed_dim]
            embeds.append(embed)

        # Stack embeds to [B, L, embed_dim]
        x = torch.stack(embeds, dim=1)  # [B, L, embed_dim]

        # Add positional encoding
        x = self.pos_encoder(x)  # [B, L, embed_dim]

        # Transformer with batch_first=True
        x = self.transformer_encoder(x)  # [B, L, embed_dim]

        # Global average pooling
        x = x.mean(dim=1)  # [B, embed_dim]

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x