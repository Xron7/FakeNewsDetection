import pandas            as pd
import networkx          as nx
import numpy             as np
import re
import emoji
import string

from tqdm  import tqdm
from utils import construct_prop_df
from utils import construct_graph

PATH      = 'twitter15/'
PROP_PATH = PATH + 'tree/'
THRESHOLDS = [0, 5, 30, 60, 180, 480, 720, 1440]

RT_UNDER_COLUMNS = []
for t in THRESHOLDS[1:]:
  RT_UNDER_COLUMNS.append(f'rts_under_{t}_min')

########################################################################################################################
# Propagation dataset functions
def get_fast_tweets(df):

  categories = pd.cut(df['time_elapsed'], bins=THRESHOLDS, right=False)

  counts = categories.value_counts().sort_index().cumsum().tolist()

  return counts


def get_half_life(df, time_total):
  return df[df['time_elapsed'] <= time_total/2].shape[0] / df.shape[0]


def count_rt_circles(df):
  return df[df['time_rt'] == 0.0].shape[0]


########################################################################################################################
# Graph Functions
def get_depth_stats(G, root):
    lengths = list(nx.single_source_shortest_path_length(G, root).values())
    return max(lengths), np.mean(lengths)


########################################################################################################################
# Application
def prop_data_pipeline(tweet_id):

  prop_df = construct_prop_df(tweet_id, PROP_PATH)

  uid = prop_df['retweeter_id'][0]

  prop_df = prop_df.iloc[1:] # exclude original post

  num_circles  = count_rt_circles(prop_df)
  num_retweets = prop_df.shape[0]

  G = construct_graph(prop_df)
  depth_max, depth_avg = get_depth_stats(G, uid)

  total_time = prop_df['time_elapsed'].max()
  avg_time   = prop_df['time_elapsed'].mean()
  half_life  = get_half_life(prop_df, total_time)

  fast_tweets = get_fast_tweets(prop_df)
  counts_dict = {}
  for i, c in enumerate(RT_UNDER_COLUMNS):
    counts_dict[c] = fast_tweets[i]

  extracted_features = {
        'poster':        uid,
        'num_rt':        num_retweets,
        'depth_max':     depth_max,
        'depth_avg':     depth_avg,
        'time_total':    total_time,
        'time_avg':      avg_time,
        'rts_half_life': half_life,
        'num_circles':   num_circles,
    }

  extracted_features.update(counts_dict)

  return extracted_features


def feature_extraction_pipeline(df):

  tqdm.pandas()

  features_df = df['tweet_id'].progress_apply(prop_data_pipeline).apply(pd.Series)
  tweet_df    = df['tweet'].progress_apply(tweet_pipeline).apply(pd.Series)
  df          = df.join(features_df)
  df          = df.join(tweet_df)

  df['day_1_perc'] = df['rts_under_1440_min']/df['num_rt']

  return df


########################################################################################################################
# Tweet functions
def tweet_pipeline(tweet):
    words = tweet.split()

    length       = len(tweet)
    num_words    = len(words)
    num_urls     = tweet.count('URL')
    num_mentions = len(re.findall(r'@\w+', tweet))
    num_hashtags = len(re.findall(r'#\w+', tweet))
    num_all_caps = len([word for word in words if word.isupper()]) - num_urls  # to not count URL
    num_emoji    = len([char for char in tweet if char in emoji.EMOJI_DATA])
    num_punc     = sum(1 for char in tweet if char in string.punctuation) - num_hashtags - num_mentions  # to not count metions and hashtags

    features = {
        'length':       length,
        'num_words':    num_words,
        'num_urls':     num_urls,
        'num_mentions': num_mentions,
        'num_hashtags': num_hashtags,
        'num_emoji':    num_emoji,
        'num_all_caps': num_all_caps,
        'num_punc':     num_punc
    }

    return features


########################################################################################################################

tweets_df = pd.read_csv(PATH + 'source_tweets.txt', sep = '\t', header = None, names = ['tweet_id', 'tweet'])
labels_df = pd.read_csv(PATH + 'label.txt', sep = ':', header = None, names = ['label', 'tweet_id'])
df        = pd.merge(tweets_df, labels_df, on="tweet_id", how="left")

# sanity_df = df.head().copy()
# sanity_df = feature_extraction_pipeline(sanity_df)
# sanity_df

df = feature_extraction_pipeline(df)

df.to_csv(PATH + 'dataset_enhanced.csv', index=False)

# TO DO
# hashtag similiarity?
# user based dataset?
# sentiment analysis?
