import sys
import torch
import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np

from sklearn.preprocessing import MinMaxScaler

from network.GATE import GATEModel
from utils        import parse_config
from config       import PATH

########################################################################################################################
# Init

config = parse_config(sys.argv[1])

hidden_dims = config["hidden_dims"]
lambda_     = config["lambda_"]
lr          = config["lr"]
epochs      = config["epochs"]

########################################################################################################################
# Data

if config["use_dummy"]:
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 1, 2],
                            [1, 0, 3, 4, 3, 2, 1]])
    structure_pairs = edge_index
else:
    # edges
    df_G = pd.read_csv('network/network.csv', header = None)
    df_G.columns = ['source', 'target', 'weight']

    # features
    df_nodes = pd.read_csv(PATH +'node_features.csv')
    df_nodes.drop(columns=['score'], inplace = True)

    # remove single retweets
    num_nodes = df_nodes.shape[0]

    mask = ~((df_nodes['num_post'] == 0) & (df_nodes['user_rt'] == 1))
    df_nodes = df_nodes[mask].reset_index(drop=True)
    print('Removing single retweeters:')
    print(f'Removed {num_nodes - df_nodes.shape[0]} nodes')
    print('----------------------------------------------------------------------------------------------')

    valid_nodes = df_nodes.loc[mask, 'user_id'].tolist()
    df_G = df_G[df_G['source'].isin(valid_nodes) & df_G['target'].isin(valid_nodes)].reset_index(drop=True)

    # they need to increment from 0
    nodes = df_nodes['user_id'].tolist()
    node2idx = {node: idx for idx, node in enumerate(nodes)}
    df_nodes.drop(columns=['user_id'], inplace = True)

    # log
    # df_nodes= np.log1p(df_nodes)

    # scale
    scaler = MinMaxScaler()
    df_nodes = scaler.fit_transform(df_nodes)

    # tensors
    x          = torch.tensor(df_nodes, dtype=torch.float)
    edge_index = torch.tensor([[node2idx[src] for src in df_G["source"]],
                               [node2idx[tgt] for tgt in df_G["target"]]], dtype=torch.long)
    structure_pairs = edge_index

########################################################################################################################
# Train Loop
dims = [df_nodes.shape[1]] + hidden_dims

model     = GATEModel(hidden_dims = dims, lambda_ = lambda_)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

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

########################################################################################################################
# Plotting

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
