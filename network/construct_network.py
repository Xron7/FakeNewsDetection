import os
import networkx as nx

from tqdm import tqdm

from config import PATH
from utils  import construct_prop_df

data_path = PATH + 'tree/'

sources = []
targets = []
weights = []

for file_name in tqdm(os.listdir(data_path), desc= 'Constructing Network'):
    
    prop_df = construct_prop_df(file_name)
    sources = prop_df['source'].tolist()
    targets = prop_df['retweeter_id'].tolist()
    weights = prop_df['time_elapsed'].tolist()

G = nx.DiGraph()

edges = zip(sources, targets, weights)
G.add_edges_from((source, retweeter, {'weight': weight}) for source, retweeter, weight in edges)

nx.write_weighted_edgelist(G, PATH + "network.csv", delimiter=',')
