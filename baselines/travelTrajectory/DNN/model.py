import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

# Multi-layer CNN Model for feature extraction and classification
class MultiLayerCNN(nn.Module):
    def __init__(self, field_dims=[], num_features = 7, embed_dim=32, num_classes=10):
        super(MultiLayerCNN, self).__init__()
        self.num_cat = len(field_dims)
        self.num_cont = num_features - self.num_cat
        self.embed_dim = embed_dim

        # Embeddings for categorical features
        self.cat_embeds = nn.ModuleList([nn.Embedding(fd, embed_dim) for fd in field_dims])

        # Conv1d for continuous features
        self.cont_embeds = nn.ModuleList(
            [nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=1) for _ in range(self.num_cont)])

        # CNN layers after embedding
        # in_channels = embed_dim
        # Layer 1: Conv1d (total_in_channels -> 32, kernel=3)
        self.conv1 = nn.Conv1d(embed_dim, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        # Layer 2: Conv1d (32 -> 64, kernel=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        # Layer 3: Conv1d (64 -> 128, kernel=3)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        # Removed pool3 to avoid output size 0 for small seq_len

        # Calculate flattened size (after two pools, seq_len // 4)
        flattened_size = 128 * (num_features // 4)
        self.fc1 = nn.Linear(flattened_size, 256)
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

        # Transpose to [B, embed_dim, L] wait, no: actually concat along dim=1 would be channels if we transpose later.
        # But since stack is [B, L=num_fields, embed_dim], to apply Conv1d, transpose to [B, embed_dim, L]
        # But if concat the embeds along dim=1, but each is [B, embed_dim], stack(dim=1) is [B, L, embed_dim], yes.
        # Then transpose(1,2) [B, embed_dim, L]? But then channels=embed_dim, but in init I set total_in_channels = (num_cat + num_cont)*embed_dim
        # Mistake.

        # If I want to concat along channel, I should make each embed [B, embed_dim, L=1], then cat dim=2 no.

        # Since each field is independent, and sequence is over the fields, so [B, L, embed_dim], transpose to [B, embed_dim, L]
        # Then in_channels = embed_dim

        # Yes, that's better. Change total_in_channels to embed_dim

        x = x.transpose(1, 2)  # [B, embed_dim, L]

        # Now apply CNN
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        # Removed pool3

        # Flatten
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x