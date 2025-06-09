import networkx as nx
import numpy as np
import pandas as pd
import torch

# Parameters
num_nodes = 677640
num_edges = 834432  # You define this

# 1. Generate random directed graph with exactly num_edges edges
G_nx = nx.gnm_random_graph(num_nodes, num_edges, directed=True, seed=42)

# 2. Remove self-loops to make it realistic (optional)
G_nx.remove_edges_from(nx.selfloop_edges(G_nx))

# 3. Get edge list as torch tensor
edges = list(G_nx.edges())
edge_index = torch.tensor(edges, dtype=torch.long).T
# 4. Create random labels and time_elapsed for each edge
num_edges = edge_index.shape[1]
labels = ["true", "false", "unverified", "non-rumor"]
np.random.seed(42)
edge_labels = np.random.choice(labels, size=num_edges)
edge_times = np.random.uniform(0, 1000, size=num_edges)

# 5. Save to CSV
df_edges = pd.DataFrame(
    {
        "source": edge_index[0].numpy(),
        "target": edge_index[1].numpy(),
        "label": edge_labels,
        "time_elapsed": edge_times,
    }
)

df_edges.to_csv("random_graph.csv", index=False)
print("Saved dummy graph with", num_edges, "edges to 'dummy_graph.csv'")

num_features = 12

np.random.seed(42)
features = np.random.exponential(scale=1.0, size=(num_nodes, num_features))


# Add score (mixed positive/negative)
score = np.linspace(-2, 2, num_nodes) + np.random.normal(0, 0.5, num_nodes)
features = np.hstack([features, score.reshape(-1, 1)])

feature_cols = [f"feat_{i}" for i in range(num_features)] + ["score"]
df_features = pd.DataFrame(features, columns=feature_cols)
df_features["user_id"] = np.arange(num_nodes)

df_features.to_csv("random_node_features.csv", index=False)
