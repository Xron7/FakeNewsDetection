import random
import networkx as nx
import numpy as np
import pandas as pd

from config import PATH

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

num_nodes = 100000
G_raw = nx.scale_free_graph(num_nodes)
G = nx.DiGraph(G_raw)
G.remove_edges_from(nx.selfloop_edges(G))
G = nx.DiGraph(((u, v) for u, v in G.edges() if u != v))  # ensure no self-loops

print(G.number_of_nodes(), G.number_of_edges())

# node features
df = pd.read_csv(PATH + "real_node_features.csv")

df_dummy = pd.DataFrame()
df_dummy["user_id"] = range(num_nodes)

for col in df.columns:
    if col != "user_id":
        df_dummy[col] = df[col].sample(n=num_nodes, replace=True).reset_index(drop=True)

# time and label
time_pool = df_dummy["user_time_rt"].dropna().values
labels = ["true", "false", "unverified", "non-rumor"]
for u, v in G.edges():
    G[u][v]["label"] = random.choice(labels)
    G[u][v]["time"] = float(np.random.choice(time_pool))

# saving
nx.write_edgelist(G, PATH + "random_network.csv", delimiter=",", data=["label", "time"])
df_dummy.to_csv(PATH + "random_node_features.csv", index=False)
