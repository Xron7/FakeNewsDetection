import sys
import networkx as nx
import pandas   as pd

from tqdm       import tqdm

from config import PATH
from utils  import construct_prop_df, plot_after_cleaning

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
total_rts = tweet_df.groupby("label").num_rt.sum().to_dict()

nodes_removed_list = []
reach_true         = []
reach_false        = []
reach_unverified   = []
reach_non_rumor    = []

step = 6000
for i in tqdm(range(1, (len(nodes_to_remove) // step) + 2), desc="Removing incrementing subsets of nodes"):
    G_temp = G.copy()
    
    subset = nodes_to_remove[:i * step]
    G_temp.remove_nodes_from(subset)

    nodes_removed = subset.copy()
    while True:
        no_incoming = [n for n in G_temp.nodes if G_temp.in_degree(n) == 0 and n not in posters]
        nodes_removed.extend(no_incoming)

        if not no_incoming:
            break

        G_temp.remove_nodes_from(no_incoming)
    
    nodes_removed_list.append((len(nodes_removed) - len(subset)) / G.number_of_nodes())

    reachabilities = {"true": 0, "false": 0, "unverified": 0, "non-rumor": 0}
    for tweet in tweet_df.itertuples():
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
    reach_true.append(reachabilities["true"])
    reach_false.append(reachabilities["false"])
    reach_unverified.append(reachabilities["unverified"])
    reach_non_rumor.append(reachabilities["non-rumor"])

plot_after_cleaning(reach_false, 'test', 'false_reachability')
