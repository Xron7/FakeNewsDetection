import pandas as pd

from tqdm import tqdm

from utils  import construct_prop_df
from config import PATH

users_df  = pd.read_csv(PATH + 'user_dataset.csv')
tweets_df = pd.read_csv(PATH + 'dataset_enhanced.csv')

users_df.set_index('user_id', inplace=True)

users_df['num_post_true']  = 0
users_df['num_post_false'] = 0
users_df['num_rt_false']   = 0
users_df['num_rt_true']    = 0

for index, row in tqdm(tweets_df.iterrows()):

    label = 'false'
    if row['label'] == 1:
        label = 'true'

    users_df.loc[row['poster'], f'num_post_{label}'] += 1

    prop_df = construct_prop_df(row['tweet_id'])
    prop_df = prop_df[1:]

    retweeters = prop_df['retweeter_id'].tolist()

    for rt in retweeters:
        users_df.loc[int(rt), f'num_rt_{label}'] += 1
    break
