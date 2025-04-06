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

# --- Demo: Small Graph with 5 nodes ---
x = torch.randn(5, 4)  # 5 nodes, 4 features each
edge_index = torch.tensor([[0, 1, 2, 3, 4, 1, 2],
                           [1, 0, 3, 4, 3, 2, 1]])  # COO format

# --- Run Encoder and Decoder ---
encoder = GATEEncoder([4, 8, 2])
H = encoder(x, edge_index)

decoder = GATEDecoder(encoder)
x_reconstructed = decoder(H, edge_index)

# --- Print ---
print("Original features:\n", x)
print("\nReconstructed features:\n", x_reconstructed)
