import torch
import torch.nn as nn
import torch.nn.functional as F


class WideDeepRecurrent(nn.Module):
    """
    Wide-Deep-Recurrent (WDR) model for Estimated Time of Arrival (ETA) prediction.
    Based on the paper "Learning to Estimate the Travel Time" by Zheng Wang et al. (KDD 2018).

    The model consists of three main components:
    - Wide: Handles memorization through linear terms and low-order feature interactions (approximated as FM-like).
    - Deep: Handles generalization through a multi-layer perceptron (MLP) on concatenated dense and embedded sparse features.
    - Recurrent: Captures sequential dependencies in road segment features using LSTM.

    The outputs from wide, deep, and recurrent are concatenated and passed through a regressor to predict the travel time.

    Hyperparameters from paper:
    - Sparse embedding dim: 20
    - Hidden dim (for wide, deep, recurrent outputs): 256
    - Deep MLP: 3 hidden layers, each 256 dim
    - LSTM cell size: 256

    Assumptions:
    - dense_dim: Number of dense (continuous) features, e.g., route length, time of day encodings.
    - sparse_vocab_sizes: List of vocabulary sizes for each sparse (categorical) feature, e.g., driver ID, road grade.
    - lstm_input_dim: Dimension of each road segment's features in the sequence (e.g., segment length, speed, etc.).
    """

    def __init__(self, dense_dim, sparse_vocab_sizes, sparse_embed_dim=20, hidden_dim=256, num_deep_layers=3,
                 lstm_input_dim=10):
        """
        Initializes the WDR model.

        Args:
            dense_dim (int): Dimension of dense features.
            sparse_vocab_sizes (list[int]): List of vocabulary sizes for each sparse feature field.
            sparse_embed_dim (int, optional): Embedding dimension for sparse features. Defaults to 20 (as per paper).
            hidden_dim (int, optional): Hidden dimension for wide, deep, and recurrent outputs. Defaults to 256 (as per paper).
            num_deep_layers (int, optional): Number of hidden layers in deep MLP. Defaults to 3 (as per paper).
            lstm_input_dim (int, optional): Input dimension for each element in the sequential features (road segments).
        """
        super(WideDeepRecurrent, self).__init__()

        # Embeddings for sparse features in the deep part (each sparse field embedded to sparse_embed_dim)
        # Input: Sparse indices (batch_size,) per field
        # Output per embedding: (batch_size, sparse_embed_dim)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, sparse_embed_dim) for vocab_size in sparse_vocab_sizes])

        # Separate embeddings for FM interactions in wide part (to model pairwise interactions)
        # Same dimensions as above for consistency
        self.fm_embed_dim = sparse_embed_dim
        self.fm_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, self.fm_embed_dim) for vocab_size in sparse_vocab_sizes])

        # Wide linear part: Input is concat of dense + flattened embedded sparse (for linear terms)
        # Input dim: dense_dim + num_sparse_fields * sparse_embed_dim
        # Output: (batch_size, hidden_dim)
        self.wide_linear_dim = dense_dim + len(sparse_vocab_sizes) * sparse_embed_dim
        self.wide_linear = nn.Linear(self.wide_linear_dim, hidden_dim)

        # Projection for FM interactions output to hidden_dim
        # FM interactions output: (batch_size, fm_embed_dim) after computation
        # Output: (batch_size, hidden_dim)
        self.wide_interact_proj = nn.Linear(self.fm_embed_dim, hidden_dim)

        # Affine transformation after combining linear + interactions in wide
        # Input: (batch_size, hidden_dim)
        # Output: (batch_size, hidden_dim)
        self.wide_affine = nn.Linear(hidden_dim, hidden_dim)

        # Deep part: MLP on concatenated dense + embedded sparse
        # Input dim: dense_dim + num_sparse_fields * sparse_embed_dim
        # Each layer: (batch_size, hidden_dim) -> ReLU -> (batch_size, hidden_dim)
        # Final output: (batch_size, hidden_dim)
        self.deep_embed_dim = dense_dim + len(sparse_vocab_sizes) * sparse_embed_dim
        deep_layers = []
        deep_layers.append(nn.Linear(self.deep_embed_dim, hidden_dim))
        for _ in range(num_deep_layers - 1):
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Linear(hidden_dim, hidden_dim))
        deep_layers.append(nn.ReLU())  # Final ReLU after last layer
        self.deep_mlp = nn.Sequential(*deep_layers)

        # Recurrent part: FC projection for sequential features, then LSTM
        # Seq FC: (batch_size, seq_len, lstm_input_dim) -> ReLU -> (batch_size, seq_len, hidden_dim)
        self.seq_fc = nn.Linear(lstm_input_dim, hidden_dim)

        # LSTM: Input (batch_size, seq_len, hidden_dim)
        # Output: (batch_size, seq_len, hidden_dim), but we take last hidden state: (batch_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Regressor: Concat wide + deep + recurrent (3 * hidden_dim) -> 1 (travel time prediction)
        # Input: (batch_size, 3 * hidden_dim)
        # Output: (batch_size, 1)
        self.regressor = nn.Linear(hidden_dim * 3, 1)

    def forward(self, dense, sparse_list, seq_features):
        """
        Forward pass of the WDR model.

        Args:
            dense (torch.Tensor): Dense features. Shape: (batch_size, dense_dim)
            sparse_list (list[torch.Tensor]): List of sparse feature indices, each shape: (batch_size,)
            seq_features (torch.Tensor): Sequential road segment features. Shape: (batch_size, seq_len, lstm_input_dim)

        Returns:
            torch.Tensor: Predicted travel times. Shape: (batch_size,)
        """
        # Deep Embeddings: Embed each sparse field
        # Each embedded_sparse[i]: (batch_size, sparse_embed_dim)
        # concat_sparse: (batch_size, num_sparse * sparse_embed_dim)
        embedded_sparse = [self.embeddings[i](sparse_list[i]) for i in range(len(sparse_list))]
        concat_sparse = torch.cat(embedded_sparse, dim=1)

        # Deep Forward: Concat dense + embedded sparse -> MLP
        # concat_deep: (batch_size, dense_dim + num_sparse * sparse_embed_dim)
        # deep_out: (batch_size, hidden_dim)
        concat_deep = torch.cat([dense, concat_sparse], dim=1)
        deep_out = self.deep_mlp(concat_deep)

        # Wide FM Linear Part: Reuse deep embeddings for linear terms (as approximation)
        # linear_inputs: (batch_size, dense_dim + num_sparse * sparse_embed_dim)
        # linear_out: (batch_size, hidden_dim)
        linear_inputs = torch.cat([dense] + embedded_sparse,
                                  dim=1)  # Note: Using deep embeds for simplicity; paper uses cross-product
        linear_out = self.wide_linear(linear_inputs)

        # Wide FM Interactions: Compute pairwise interactions (FM style)
        # fm_embeds: list of (batch_size, fm_embed_dim)
        # sum_square: (batch_size, fm_embed_dim)
        # square_sum: (batch_size, fm_embed_dim)
        # interactions: (batch_size, fm_embed_dim)
        # interact_out: (batch_size, hidden_dim)
        fm_embeds = [self.fm_embeddings[i](sparse_list[i]) for i in range(len(sparse_list))]
        sum_square = (torch.sum(torch.stack(fm_embeds, dim=1), dim=1)) ** 2
        square_sum = torch.sum(torch.stack([e ** 2 for e in fm_embeds], dim=1), dim=1)
        interactions = 0.5 * (sum_square - square_sum)
        interact_out = self.wide_interact_proj(interactions)

        # Wide Combine: Linear + interactions -> affine
        # wide_out: (batch_size, hidden_dim)
        wide_out = self.wide_affine(linear_out + interact_out)

        # Recurrent Forward: Project seq -> LSTM -> last hidden
        # seq_projected: (batch_size, seq_len, hidden_dim)
        # lstm_out: (batch_size, seq_len, hidden_dim)
        # recurrent_out: (batch_size, hidden_dim) [last h_n]
        seq_projected = F.relu(self.seq_fc(seq_features))
        lstm_out, (h_n, c_n) = self.lstm(seq_projected)
        recurrent_out = h_n.squeeze(0)  # h_n shape: (1, batch_size, hidden_dim) -> squeeze to (batch_size, hidden_dim)

        # Regressor: Concat all three outputs -> linear to scalar
        # combined: (batch_size, 3 * hidden_dim)
        # pred: (batch_size, 1) -> squeeze to (batch_size,)
        combined = torch.cat([wide_out, deep_out, recurrent_out], dim=1)
        pred = self.regressor(combined)
        return pred.squeeze(1)