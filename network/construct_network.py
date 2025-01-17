import os
import networkx as nx

from tqdm import tqdm

from config import PATH

data_path = PATH + 'tree/'

sources = []
targets = []
weights = []

for file_name in tqdm(os.listdir(data_path), desc= 'Constructing Network'):
    with open(data_path + file_name, 'r') as file:

        # first line for the poster
        first_line = file.readline().strip()
        root, _    = first_line.split('->')

        root = eval(root)

        # check if there is time error
        time_shift = 0
        if root[0]!= 'ROOT':
            print(f'Detected time issue for {file_name}')
            time_shift = -1 * float(root[2])

        for i, line in enumerate(file):
            source, retweet = line.split('->')
            source  = eval(source)
            retweet = eval(retweet)

            if source[0]!='ROOT':
                sources.append(source[0])
                targets.append(retweet[0])
                weights.append(round(float(retweet[2]) - float(source[2]) + time_shift, 2))

G = nx.DiGraph()

edges = zip(sources, targets, weights)
G.add_edges_from((source, retweeter, {'weight': weight}) for source, retweeter, weight in edges)

nx.write_weighted_edgelist(G, "network.csv", delimiter=',')
