import torch
import torch.nn as nn
from torch_geometric.utils import softmax

# --- GATE Attention Layer ---
class GATEAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.v0 = nn.Parameter(torch.Tensor(out_dim, 1))
        self.v1 = nn.Parameter(torch.Tensor(out_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.v0)
        nn.init.xavier_uniform_(self.v1)

    def forward(self, x, edge_index):
        h = self.W(x)
        alpha_0 = (h @ self.v0).squeeze(-1)
        alpha_1 = (h @ self.v1).squeeze(-1)

        row, col = edge_index
        scores = torch.sigmoid(alpha_0[row] + alpha_1[col])
        attn = softmax(scores, row)

        out = torch.zeros_like(h)
        out.index_add_(0, row, h[col] * attn.unsqueeze(-1))
        return out, attn

# --- GATE Encoder ---
class GATEEncoder(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        self.layers = nn.ModuleList([
            GATEAttentionLayer(hidden_dims[i], hidden_dims[i+1])
            for i in range(len(hidden_dims) - 1)
        ])

    def forward(self, x, edge_index):
        self.attentions = []
        for layer in self.layers:
            x, attn = layer(x, edge_index)
            self.attentions.append(attn)
        return x

# --- Decoder Layer ---
class GATEDecoderLayer(nn.Module):
    def __init__(self, encoder_layer, in_dim, out_dim):
        super().__init__()
        self.encoder_layer = encoder_layer
        self.W_T = nn.Parameter(encoder_layer.W.weight)  # Transpose of encoder weights
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, h, edge_index, attn):
        h = h @ self.W_T  # Apply transposed weights
        row, col = edge_index
        out = torch.zeros(h.size(0), self.out_dim, device=h.device)
        out.index_add_(0, row, h[col] * attn.unsqueeze(-1))
        return out

# --- GATE Decoder ---
class GATEDecoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.layers = nn.ModuleList([
            GATEDecoderLayer(layer, encoder.layers[-1 - i].W.out_features, encoder.layers[-1 - i].W.in_features) 
            for i, layer in enumerate(reversed(encoder.layers))
        ])

    def forward(self, h, edge_index):
        for i, layer in enumerate(self.layers):
            attn = self.encoder.attentions[-1 - i]
            h = layer(h, edge_index, attn)
        return h

class GATEModel(nn.Module):
    def __init__(self, hidden_dims, lambda_):
        super().__init__()
        self.lambda_ = lambda_
        self.encoder = GATEEncoder(hidden_dims)
        self.decoder = GATEDecoder(self.encoder)

    def forward(self, x, edge_index, structure_pairs):
        """
        structure_pairs: (row, col) indices of edges to reconstruct structure from
        """
        # Encode
        h = self.encoder(x, edge_index)

        # Decode
        x_recon = self.decoder(h, edge_index)

        # Feature reconstruction loss (L2)
        feature_loss = torch.norm(x - x_recon, p=2)

        # Structure reconstruction loss (inner product of pairs)
        row, col = structure_pairs  # same shape as edge_index
        h_i = h[row]
        h_j = h[col]
        dot_product = torch.sum(h_i * h_j, dim=1)
        structure_loss = -torch.log(torch.sigmoid(dot_product) + 1e-8).sum()

        total_loss = feature_loss + self.lambda_ * structure_loss
        return total_loss, h, x_recon


model = GATEModel(hidden_dims=[4, 8, 2], lambda_=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Dummy graph input
x = torch.randn(5, 4)
edge_index = torch.tensor([[0, 1, 2, 3, 4, 1, 2],
                           [1, 0, 3, 4, 3, 2, 1]])

structure_pairs = edge_index  # usually same as edge_index

# Training step
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    loss, h, x_recon = model(x, edge_index, structure_pairs)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")
