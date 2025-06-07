import random
import numpy as np
import networkx as nx
import pandas as pd

from utils import PATH

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

G = nx.DiGraph()
G = nx.read_edgelist(
    PATH + "network.csv",
    delimiter=",",
    nodetype=int,
    data=(("weight", float),),
    create_using=nx.DiGraph(),
)

in_degree = list(d for _, d in G.in_degree())
out_degree = list(d for _, d in G.out_degree())

G_dist = nx.directed_configuration_model(in_degree, out_degree)

G_dist = nx.DiGraph(G_dist)
G_dist.remove_edges_from(nx.selfloop_edges(G_dist))

# Replace dummy node ids with real user_ids
features_df = pd.read_csv(PATH + "node_features.csv")
idx2node = {idx: node for idx, node in enumerate(features_df["user_id"].tolist())}
G_dist = nx.relabel_nodes(G_dist, idx2node)

nx.write_edgelist(G_dist, PATH + "same_dist_network.csv", delimiter=",", data=False)
