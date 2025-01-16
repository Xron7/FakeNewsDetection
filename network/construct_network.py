import os

from tqdm import tqdm

from config import PATH
from utils  import construct_prop_df

data_path = PATH + 'tree/'

for file_name in tqdm(os.listdir(data_path), desc= 'Constructing Network'):
    with open(data_path + '752965545528528898.txt', 'r') as file:

        # first line for the poster
        first_line   = file.readline().strip()
        root, poster = first_line.split('->')

        root   = eval(root)
        poster = eval(poster)

        poster = poster[0]

        # check if there is time error
        time_shift = 0
        if root[0]!= 'ROOT':
            print(f'Detected time issue for {file_name}')
            time_shift = -1 * float(root[2])

        # separate handling to remove the ROOT
        if time_shift:
            pass
        else:
            pass
        # for i, line in enumerate(file):
        #     source, retweet = line.split('->')
        #     source  = eval(source)
        #     retweet = eval(retweet)
        #     print(source, retweet)
        #
        #     poster = retweet[0]
        #
        #     break

    break

