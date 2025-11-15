import torch
import torch.nn as nn
import torch.nn.functional as F


class WideDeepRecurrent(nn.Module):
    def __init__(self, dense_dim, sparse_vocab_sizes, sparse_embed_dim=20, hidden_dim=256, num_deep_layers=3,
                 lstm_input_dim=10):
        super(WideDeepRecurrent, self).__init__()

        # Embeddings for sparse features (shared for deep, separate for wide FM)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, sparse_embed_dim) for vocab_size in sparse_vocab_sizes])

        # For wide FM
        self.fm_embed_dim = sparse_embed_dim
        self.fm_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, self.fm_embed_dim) for vocab_size in sparse_vocab_sizes])

        # Linear for wide
        self.wide_linear_dim = dense_dim + len(sparse_vocab_sizes) * sparse_embed_dim
        self.wide_linear = nn.Linear(self.wide_linear_dim, hidden_dim)

        # Interaction projection
        self.wide_interact_proj = nn.Linear(self.fm_embed_dim, hidden_dim)

        # Wide affine
        self.wide_affine = nn.Linear(hidden_dim, hidden_dim)  # perhaps combine

        # Deep part
        self.deep_embed_dim = dense_dim + len(sparse_vocab_sizes) * sparse_embed_dim
        self.deep_mlp = nn.Sequential(
            nn.Linear(self.deep_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )  # 3 hidden layers

        # Recurrent
        self.seq_fc = nn.Linear(lstm_input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Regressor
        self.regressor = nn.Linear(hidden_dim * 3, 1)

    def forward(self, dense, sparse_list, seq_features):
        # dense: (batch, dense_dim)
        # sparse_list: list of (batch,) tensors, one per sparse field
        # seq_features: (batch, seq_len, lstm_input_dim)

        # Embed for deep
        embedded_sparse = [self.embeddings[i](sparse_list[i]) for i in range(len(sparse_list))]
        concat_sparse = torch.cat(embedded_sparse, dim=1)

        # Deep
        concat_deep = torch.cat([dense, concat_sparse], dim=1)
        deep_out = self.deep_mlp(concat_deep)

        # Wide FM
        # Linear part
        linear_inputs = torch.cat([dense] + embedded_sparse, dim=1)  # use deep embeds for linear
        linear_out = self.wide_linear(linear_inputs)

        # FM interactions
        fm_embeds = [self.fm_embeddings[i](sparse_list[i]) for i in range(len(sparse_list))]
        sum_square = (torch.sum(torch.stack(fm_embeds, dim=1), dim=1)) ** 2
        square_sum = torch.sum(torch.stack([e ** 2 for e in fm_embeds], dim=1), dim=1)
        interactions = 0.5 * (sum_square - square_sum)
        interact_out = self.wide_interact_proj(interactions)

        wide_out = self.wide_affine(linear_out + interact_out)

        # Recurrent
        seq_projected = F.relu(self.seq_fc(seq_features))
        lstm_out, (h_n, c_n) = self.lstm(seq_projected)
        recurrent_out = h_n.squeeze(0)

        # Combine
        combined = torch.cat([wide_out, deep_out, recurrent_out], dim=1)
        pred = self.regressor(combined)
        return pred.squeeze(1)