from GATE import GATEModel
import torch

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