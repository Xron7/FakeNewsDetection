"""
Removes the nodes from the network and calculates the reachability.
arg1: the network title e.g. real
arg2: the nodes to remove
"""

import sys
import networkx as nx
import pandas as pd
from tqdm import tqdm

from config import PATH


def count_retweets_per_label(G):
    labels = ["true", "false", "unverified", "non-rumor"]
    total_rts = {label: 0 for label in labels}
    for _, _, data in tqdm(G.edges(data=True), desc="Counting Retweets per Label"):
        label = data["label"]
        total_rts[label] += 1
    return total_rts


title = sys.argv[1]

########################################################################################################################
# Read Inputs
G = nx.DiGraph()
G = nx.read_edgelist(
    PATH + f"/{title}_network.csv",
    delimiter=",",
    nodetype=int,
    data=(
        ("label", str),
        ("weight", float),
    ),
    create_using=nx.DiGraph(),
)

with open(sys.argv[2], "r") as f:
    nodes_to_remove = [int(line.strip()) for line in f if line.strip()]
nodes_to_remove = nodes_to_remove[:1000]  # limit to 1000 nodes for performance
features_df = pd.read_csv(PATH + f"/{title}_node_features.csv")
posters = features_df[features_df["num_post"] != 0]["user_id"].tolist()

# total retweets per label
total_rts = count_retweets_per_label(G)

########################################################################################################################
# Remove nodes and the nodes that depend on them
num_nodes_og = G.number_of_nodes()
G.remove_nodes_from(nodes_to_remove)

nodes_removed = nodes_to_remove.copy()
while True:
    no_incoming = [n for n in G.nodes if G.in_degree(n) == 0 and n not in posters]
    nodes_removed.extend(no_incoming)

    if not no_incoming:
        break

    G.remove_nodes_from(no_incoming)

num_removed = len(nodes_removed)
print(f"{num_removed} nodes removed ({100 * num_removed / num_nodes_og:.4f}%)")
print("---------------------------------------------------------------")

########################################################################################################################
# Calculate reachability after cleaning

reachabilities = count_retweets_per_label(G)

reachabilities = {k: reachabilities[k] / total_rts[k] for k in reachabilities}
print("Reachability after node removal:")
for label, r in reachabilities.items():
    print(f"{label}: {r:.4f}")
