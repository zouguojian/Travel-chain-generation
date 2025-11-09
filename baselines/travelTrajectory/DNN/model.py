import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F


# Multi-layer MLP Model for feature extraction and classification
class MultiLayerMLP(nn.Module):
    def __init__(self, field_dims=[], num_features = 7, embed_dim=32, num_classes=10):
        super(MultiLayerMLP, self).__init__()
        self.num_cat = len(field_dims)
        self.num_cont = num_features - self.num_cat
        self.embed_dim = embed_dim

        # Embeddings for categorical features
        self.cat_embeds = nn.ModuleList([nn.Embedding(fd, embed_dim) for fd in field_dims])

        # Conv1d for continuous features
        self.cont_embeds = nn.ModuleList(
            [nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=1) for _ in range(self.num_cont)])

        # MLP layers after embedding
        flattened_size = num_features * embed_dim
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

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

        # Flatten for MLP
        x = x.view(x.size(0), -1)  # [B, L * embed_dim]

        # Apply MLP layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x