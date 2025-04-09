import sys
import torch
import matplotlib.pyplot as plt

from network.GATE import GATEModel
from utils        import parse_config

config = parse_config(sys.argv[1])
hidden_dims = config["hidden_dims"]
lambda_     = config["lambda_"]
lr          = config["lr"]
epochs      = config["epochs"]

model     = GATEModel(hidden_dims = hidden_dims, lambda_ = lambda_)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

if config["use_dummy"]:
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 1, 2],
                            [1, 0, 3, 4, 3, 2, 1]])
    structure_pairs = edge_index

losses = []

model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    loss, h, x_recon = model(x, edge_index, structure_pairs)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

plt.figure(figsize=(8, 5))
plt.plot(losses, label="Structure Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GATE Training Loss Curve")
params_str = f"λ={'d'}, epochs = {epochs}"
params_str = (
    f"λ = {lambda_}\n"
    f"lr = {lr}\n"
    f"epochs = {epochs}\n"
    f"dims = {hidden_dims}\n"
)
plt.plot([], [], ' ', label=params_str)
plt.legend()
plt.grid(True)

plt.savefig(f"{lambda_}_{lr}_{epochs}_{hidden_dims}.png", dpi=300)
