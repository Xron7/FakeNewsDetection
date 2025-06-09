import random
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm

from config import PATH, MAX_RT_SCORE, ALPHA, WEIGHTS

###########################################################################################################
# Setup

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def create_fixed_sum_dist(df, fixed_sum, col_name):
    num_non_zero = np.random.randint(1, fixed_sum + 1)
    breaks = sorted(
        np.random.choice(range(1, fixed_sum), num_non_zero - 1, replace=False)
    )
    parts = [a - b for a, b in zip(breaks + [fixed_sum], [0] + breaks)]

    df[col_name] = 0
    indices = np.random.choice(df.shape[0], num_non_zero, replace=False)

    df.loc[indices, col_name] = parts

    return features_df


###########################################################################################################
# Dummy with same distribution

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

num_rt = G_dist.number_of_edges()
print("Dummy graph with same distribution created:")
print(f"Number of nodes: {G_dist.number_of_nodes()}")
print(f"Number of edges: {num_rt}")
print("----------------------------------------------------------------")

###########################################################################################################
# Adding the labels
tweet_df = pd.read_csv(PATH + "dataset_enhanced.csv")
rt_per_label = tweet_df.groupby("label")["num_rt"].sum()
rt_per_label = rt_per_label / rt_per_label.sum()

print("Retweets per label percentage:")
print("-----------------------------------")
print(rt_per_label)
print("----------------------------------------------------------------")
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
    G_dist, PATH + "dummy_network.csv", delimiter=",", data=["label", "time"]
)

###########################################################################################################
# Dummy node features
nodes = list(G_dist.nodes())
features = []
for n in tqdm(nodes, desc="Constructing Node Features"):
    out_deg = G_dist.out_degree(n)
    in_deg = G_dist.in_degree(n)

    label_counts = {label: 0 for label in labels}
    avg_time = 0.0
    score = 0.0
    for _, _, edge_data in G_dist.in_edges(n, data=True):
        time = edge_data["time"]
        label = edge_data["label"]

        label_counts[label] += 1
        avg_time += time  # Assumption: time = avg of rts per label
        score += WEIGHTS[label] * MAX_RT_SCORE * np.exp(-ALPHA * time / 60)

    avg_time /= max(1, in_deg)
    row = {
        "user_id": n,
        "user_rt": in_deg,
        "user_time_rt": avg_time,
        "score": score,
    }

    for label in labels:
        row[f"num_rt_{label}"] = label_counts[label]

    features.append(row)

features_df = pd.DataFrame(features)

#######################################################
# num_post and rt_total
# Random distributions that ads up to # tweets and # edges

features_df = create_fixed_sum_dist(features_df, tweet_df.shape[0], "num_post")
features_df = create_fixed_sum_dist(features_df, num_rt, "rt_total")
#######################################################
# num_post per label
# Split the posts based on the label distribution
post_per_label = features_df["num_post"].values[:, None] * percs
post_per_label_int = np.floor(post_per_label).astype(int)
residuals = features_df["num_post"].values - post_per_label_int.sum(axis=1)

# Assumption: The residuals are added to the most frequent label
max_label_idx = np.argmax(percs)
post_per_label_int[:, max_label_idx] += residuals
for j, label in enumerate(labels):
    features_df[f"num_post_{label}"] = post_per_label_int[:, j]

#######################################################
# score for posts
for label in labels:
    features_df["score"] += features_df[f"num_post_{label}"] * WEIGHTS[label]

#######################################################
# verification and saving
print("----------------------------------------------------------------")
print(f"num_post total: {features_df['num_post'].sum()}")
print(f"num_post total: {features_df['rt_total'].sum()}")
print("----------------------------------------------------------------")
print("Score based on posts check:")
print("----------------------------------------------------------------")
print(
    features_df[features_df["num_post"] != 0][
        [f"num_post_{label}" for label in labels] + ["score", "num_post"]
    ].head()
)
print("----------------------------------------------------------------")
print("Score based on retweets check:")
print("----------------------------------------------------------------")
print(
    features_df[features_df["user_rt"] != 0][
        [f"num_rt_{label}" for label in labels]
        + ["score", "user_rt", "user_time_rt", "num_post"]
    ].head()
)

features_df.to_csv(PATH + "dummy_node_features.csv", index=False)
