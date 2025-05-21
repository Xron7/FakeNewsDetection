import sys
import networkx as nx

from config import PATH

########################################################################################################################
# Read Inputs

G = nx.DiGraph()
G = nx.read_edgelist(PATH + "network.csv",
                     delimiter=",", nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())

with open(sys.argv[1], "r") as f:
    nodes_to_remove = [int(line.strip()) for line in f if line.strip()]

########################################################################################################################
# Remove nodes and the nodes that depend on them

num_nodes_og = G.number_of_nodes()
G.remove_nodes_from(nodes_to_remove)

total_nodes_removed = len(nodes_to_remove)
while True:
    no_incoming = [n for n in G.nodes if G.in_degree(n) == 0]
    total_nodes_removed += len(no_incoming)

    if not no_incoming:
        break

    G.remove_nodes_from(no_incoming)

print(f'{total_nodes_removed}/{num_nodes_og} nodes removed')
