import sys
import networkx as nx
import pandas   as pd

from tqdm import tqdm

from config import PATH
from utils  import construct_prop_df

########################################################################################################################
# Read Inputs
G = nx.DiGraph()
G = nx.read_edgelist(PATH + "network.csv",
                     delimiter=",", nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())

with open(sys.argv[1], "r") as f:
    nodes_to_remove = [int(line.strip()) for line in f if line.strip()]

tweet_df = pd.read_csv(PATH + "dataset_enhanced.csv")
posters = tweet_df.poster.tolist()

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
print(f'{num_removed} nodes removed ({100*num_removed/num_nodes_og:.4f}%)')
print('---------------------------------------------------------------')

########################################################################################################################
# Compare reachability before and after
total_rts = tweet_df.groupby("label").num_rt.sum().to_dict()

reachabilities = {"true": 0, "false": 0, "unverified": 0, "non-rumor": 0}
for tweet in tqdm(tweet_df.itertuples(), desc="Computing reachability"):
    label    = tweet.label
    tweet_id = tweet.tweet_id
    
    if tweet.poster in nodes_removed:
        reachabilities[label] += 0
        continue

    prop_df = construct_prop_df(tweet_id, logging=False)
    nodes_removed_str = {str(x) for x in nodes_removed}
    prop_df = prop_df[~(prop_df.source.isin(nodes_removed_str) | prop_df.retweeter_id.isin(nodes_removed_str))]
    reachabilities[label] += prop_df.shape[0]

reachabilities = {k: reachabilities[k] / total_rts[k] for k in reachabilities}
print('Reachability after node removal:')
for l, r in reachabilities.items():
    print(f'{l}: {r:.4f}')
