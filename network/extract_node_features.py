###
# Creates the user/node features for use in the GAT
###

import pandas as pd
import numpy  as np

from tqdm import tqdm

from utils  import get_retweet_stats
from config import PATH

########################################################################################################################
# for scoring users
WEIGHTS      = {'true': 1, 'non-rumor': 1, 'unverified': 0, 'false': -1}
MAX_RT_SCORE = 0.8
ALPHA        = 1
########################################################################################################################
# read datasets
users_df  = pd.read_csv(PATH + 'user_dataset.csv', index_col=0)
tweets_df = pd.read_csv(PATH + 'dataset_enhanced.csv')

########################################################################################################################
# new column creation
types  = ['post', 'rt']
labels = tweets_df['label'].unique()

for t in types:
    for l in labels:
        users_df[f'num_{t}_{l}'] = 0

users_df['score'] = 0.0

########################################################################################################################
# iteration over tweets
for index, row in tqdm(tweets_df.iterrows(), desc = 'Constructing Node Features'):

    label = row['label']
    w     = WEIGHTS[label]

    users_df.loc[row['poster'], [f'num_post_{label}', 'score']] += [1, w]

    rt_df = get_retweet_stats(row['tweet_id'])

    for rter, t in zip(rt_df['retweeter_id'], rt_df['time_elapsed']):
        users_df.loc[int(rter), [f'num_rt_{label}', 'score']] += [1, w * MAX_RT_SCORE * np.exp(-ALPHA * max(0,t) / 60)]

users_df.to_csv(PATH + 'node_features.csv')
