###
# Creates the user/node features for use in the GAT
###

import pandas as pd

from tqdm import tqdm

from utils  import construct_prop_df, get_retweet_stats
from config import PATH

users_df  = pd.read_csv(PATH + 'user_dataset.csv', index_col=0)
tweets_df = pd.read_csv(PATH + 'dataset_enhanced.csv')

types = ['post', 'rt']
labels = tweets_df['label'].unique()

for t in types:
    for l in labels:
        users_df[f'num_{t}_{l}'] = 0

for index, row in tqdm(tweets_df.iterrows(), desc = 'Constructing Node Features'):

    label = row['label']

    users_df.loc[row['poster'], f'num_post_{label}'] += 1

    prop_df = construct_prop_df(row['tweet_id'])
    prop_df = prop_df[1:]

    retweeters = get_retweet_stats(row['tweet_id'])['retweeter_id'].tolist()

    for rt in retweeters:
        users_df.loc[int(rt), f'num_rt_{label}'] += 1

users_df.to_csv(PATH + 'node_features.csv')
