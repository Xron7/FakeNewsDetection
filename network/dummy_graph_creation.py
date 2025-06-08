import random
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm

from utils import PATH

###########################################################################################################
# Dummy with same distribution
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
print("Dummy graph with same distribution created:")
print("Number of nodes:", G_dist.number_of_nodes())
print("Number of edges:", G_dist.number_of_edges())
print("----------------------------------------------------------------")

###########################################################################################################
# Adding the labels
tweet_df = pd.read_csv(PATH + "dataset_enhanced.csv")
rt_per_label = tweet_df.groupby("label")["num_rt"].sum()
rt_per_label = rt_per_label / rt_per_label.sum()

print("Retweets per label percentage:")
print("-----------------------------------")
print(rt_per_label)
labels = rt_per_label.index.tolist()
percs = rt_per_label.values.tolist()

###########################################################################################################
# Adding the times
time_per_label = tweet_df.groupby("label")["time_avg"].mean()
time_per_label = time_per_label.to_dict()

###########################################################################################################
# Create and save the graph
for u, v in tqdm(G_dist.edges(), desc="Adding labels and times"):
    label = np.random.choice(labels, p=percs)
    G_dist[u][v]["label"] = label
    G_dist[u][v]["time"] = time_per_label[label]

nx.write_edgelist(
    G_dist, PATH + "same_dist_network.csv", delimiter=",", data=["label", "time"]
)

###########################################################################################################
# Dummy node features
# num_post_unverified,num_post_non-rumor,num_post_true,num_post_false,score,rt_total
nodes = list(G_dist.nodes())
features = []
for n in tqdm(nodes, desc="Constructing Node Features"):
    out_deg = G_dist.out_degree(n)
    in_deg = G_dist.in_degree(n)
    num_post = max(0, out_deg - in_deg)  # assumption

    label_counts = {label: 0 for label in labels}
    avg_time = 0.0
    for _, _, edge_data in G_dist.in_edges(n, data=True):
        avg_time += edge_data["time"]
        label = edge_data["label"]
        label_counts[label] += 1

    avg_time /= max(1, in_deg)
    row = {
        "user_id": n,
        "user_rt": in_deg,
        "num_post": num_post,
        "user_time_rt": avg_time,
    }

    for label in labels:
        row[f"num_rt_{label}"] = label_counts[label]

    features.append(row)

features_df = pd.DataFrame(features)
print(features_df.head())
