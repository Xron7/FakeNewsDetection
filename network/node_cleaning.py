import sys
import networkx as nx

from config import PATH

########################################################################################################################
# Read Inputs

G = nx.DiGraph()
G = nx.read_edgelist(PATH + "network.csv", delimiter=",", nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())

with open(sys.argv[1], "r") as f:
    nodes_to_remove = [int(line.strip()) for line in f if line.strip()]

num_nodes_og = G.number_of_nodes()
G.remove_nodes_from(nodes_to_remove)
print(f'{len(nodes_to_remove)}/{num_nodes_og} nodes removed')

    # while True:
    #     # Find all nodes with no incoming edges
    #     no_incoming = [n for n in G.nodes if G.in_degree(n) == 0]

    #     # If no such nodes remain, we're done
    #     if not no_incoming:
    #         break

    #     # Remove them
    #     G.remove_nodes_from(no_incoming)

    # return G
