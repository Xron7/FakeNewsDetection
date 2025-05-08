import sys
import torch
import pandas            as pd
import numpy             as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from network.GATE import GATEModel
from utils        import parse_config, log_transform, calculate_feature_loss, calculate_structure_loss, plot_loss
from config       import PATH

########################################################################################################################
# Init

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

config = parse_config(sys.argv[1])

hidden_dims      = config["hidden_dims"]
lambda_          = config["lambda_"]
lr               = config["lr"]
epochs           = config["epochs"]
remove_rt_thresh = config["remove_rt_thresh"]
scale            = config["scale"]

########################################################################################################################
# Data

if config["use_dummy"]:
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 1, 2],
                            [1, 0, 3, 4, 3, 2, 1]])
    structure_pairs = edge_index

    input_dim = 4
else:
    # edges
    df_G = pd.read_csv(PATH + 'network.csv', header = None)
    df_G.columns = ['source', 'target', 'weight']

    # features
    df_nodes = pd.read_csv(PATH +'node_features.csv')
    df_nodes.drop(columns=['score'], inplace = True)

    # remove few retweets
    if remove_rt_thresh:
        num_nodes = df_nodes.shape[0]

        mask = ~((df_nodes['num_post'] == 0) & (df_nodes['user_rt'] <= remove_rt_thresh))
        df_nodes = df_nodes[mask].reset_index(drop=True)

        removed_percent = (num_nodes - df_nodes.shape[0]) / num_nodes * 100

        print('Removing single retweeters:')
        print(f'Removed {num_nodes - df_nodes.shape[0]} nodes ({removed_percent:.2f}%)')
        print('----------------------------------------------------------------------------------------------')

        valid_nodes = df_nodes.loc[mask, 'user_id'].tolist()
        df_G = df_G[df_G['source'].isin(valid_nodes) & df_G['target'].isin(valid_nodes)].reset_index(drop=True)

    # they need to increment from 0
    nodes = df_nodes['user_id'].tolist()
    node2idx = {node: idx for idx, node in enumerate(nodes)}
    df_nodes.drop(columns=['user_id'], inplace = True)

    # log
    if config["log"]:
        df_nodes= log_transform(df_nodes)

    # scale
    if config["scale"] == "minmax":
        scaler = MinMaxScaler()
        df_nodes = scaler.fit_transform(df_nodes)
    elif config["scale"] == "standard":
        scaler = StandardScaler()
        df_nodes = scaler.fit_transform(df_nodes)
    else:
        df_nodes = df_nodes.values

    # tensors
    x          = torch.tensor(df_nodes, dtype=torch.float)
    edge_index = torch.tensor([[node2idx[src] for src in df_G["source"]],
                               [node2idx[tgt] for tgt in df_G["target"]]], dtype=torch.long)
    structure_pairs = edge_index

    input_dim = df_nodes.shape[1]

########################################################################################################################
# Train Loop
dims = [input_dim] + hidden_dims

model     = GATEModel(hidden_dims = dims, lambda_ = lambda_)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

losses           = []
feature_losses   = []
structure_losses = []

model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    loss, h, x_recon = model(x, edge_index, structure_pairs)
    loss.backward()
    optimizer.step()

    total_loss     = loss.item()
    feature_loss   = calculate_feature_loss(x, x_recon).item()
    structure_loss = calculate_structure_loss(h, structure_pairs).item()
    losses.append(total_loss)
    feature_losses.append(feature_loss)
    structure_losses.append(structure_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {total_loss:12.4f} | Feature Loss: {feature_loss:12.4f} | Structure Loss: {structure_loss:12.4f}")

########################################################################################################################
# Validation

print('----------------------------------------------------------------------------------------------')
print("x mean/std:", x.mean().item(), x.std().item())
print("x_recon mean/std:", x_recon.mean().item(), x_recon.std().item())
print("h mean/std:", h.mean().item(), h.std().item())
num_zeros = torch.sum(h == 0).item()
print(f"{num_zeros}/{h.numel()} elements are zero")

########################################################################################################################
# Plotting

plot_loss(losses, "Total", lambda_, lr, epochs, dims)
plot_loss(feature_losses, "Feature", lambda_, lr, epochs, dims)
plot_loss(structure_losses, "Structure", lambda_, lr, epochs, dims)

########################################################################################################################
# Save the embeddings
idx2node = {idx: node for node, idx in node2idx.items()}
col_names = [f"emb_{i}" for i in range(input_dim)]

embeddings_df = pd.DataFrame(h.detach().cpu().numpy(), columns=col_names)
embeddings_df["user_id"] = embeddings_df.index.map(idx2node)
embeddings_df.to_csv(PATH + "node_embeddings.csv", index=False)
