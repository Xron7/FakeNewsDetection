import pandas as pd

from tqdm import tqdm

from utils  import construct_prop_df
from config import PATH

tweets_df = pd.read_csv(PATH + 'dataset_enhanced.csv')
tweets_list = tweets_df['tweet_id'].tolist()

def create_user_df(prop_df):

    user_df = (
        prop_df.groupby("retweeter_id").size().reset_index()
    )

    user_df.columns = ['user_id', 'num_rt']
    user_df['num_post'] = 0

    poster_idx = user_df[user_df["user_id"] == prop_df.iloc[0]["retweeter_id"]].index.tolist()[0]
    user_df.at[poster_idx, "num_rt"] -= 1
    user_df.at[poster_idx, "num_post"] = 1

    return user_df


df = pd.DataFrame()
for tweet in tqdm(tweets_list, desc="Generating user df"):
    prop_df = construct_prop_df(tweet)
    user_df = create_user_df(prop_df)

    df = pd.concat([df, user_df], ignore_index=True)

print('Grouping by...')
df = (
        df.groupby("user_id", as_index=False)
        .agg({
            "num_rt": "sum",
            "num_post": "sum"
        })
    )

print('Done!')

df.to_csv(PATH + 'user_dataset.csv', index=False)
