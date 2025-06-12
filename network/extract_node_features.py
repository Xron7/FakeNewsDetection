###
# Creates the user/node features for use in the GNNs
###

import pandas as pd
import numpy as np

from tqdm import tqdm

from utils import get_retweet_stats
from config import PATH, MAX_RT_SCORE, ALPHA, WEIGHTS

########################################################################################################################
# read datasets
users_df = pd.read_csv(PATH + "user_dataset.csv", index_col=0)
tweets_df = pd.read_csv(PATH + "dataset_enhanced.csv")

########################################################################################################################
# new column creation
types = ["post", "rt"]
labels = tweets_df["label"].unique()

for t in types:
    for label in labels:
        users_df[f"num_{t}_{label}"] = 0

users_df["score"] = 0.0
users_df["rt_total"] = 0

########################################################################################################################
# iteration over tweets
for index, row in tqdm(tweets_df.iterrows(), desc="Constructing Node Features"):
    label = row["label"]
    w = WEIGHTS[label]

    users_df.loc[row["poster"], [f"num_post_{label}", "score"]] += [1, w]

    rt_df = get_retweet_stats(row["tweet_id"])

    users_df.loc[row["poster"], "rt_total"] += rt_df.shape[0]

    for rter, t in zip(rt_df["retweeter_id"], rt_df["time_elapsed"]):
        users_df.loc[int(rter), [f"num_rt_{label}", "score"]] += [
            1,
            w * MAX_RT_SCORE * np.exp(-ALPHA * max(0, t) / 60),
        ]

users_df.to_csv(PATH + "node_features.csv")
