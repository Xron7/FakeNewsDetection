import pandas            as pd
import networkx          as nx
import matplotlib.pyplot as plt
import seaborn           as sns
import numpy             as np

from config import PATH

def construct_prop_df(tweet_id):
  propagation_path = PATH + 'tree/' + f'{tweet_id}' + '.txt'

  sources    = []
  retweeters = []
  times      = []
  rt_times   = []
  with open(propagation_path, 'r') as file:
    for i, line in enumerate(file):

      source, retweet = line.split('->')
      source  = eval(source)
      retweet = eval(retweet)

      sources.append(source[0])
      retweeters.append(retweet[0])
      rt_times.append(max(0, float(retweet[2]) - float(source[2]))) # max for handling retweet circles
      times.append(float(retweet[2]))

    propagation_df = pd.DataFrame({
    'source':       sources,
    'retweeter_id': retweeters,
    'time_elapsed': times,
    'time_rt':      rt_times
})
    return propagation_df

def construct_graph(df):
  G = nx.DiGraph()

  sources    = df['source'].tolist()
  retweeters = df['retweeter_id'].tolist()
  times      = df['time_elapsed'].tolist()

  edges = zip(sources, retweeters, times)
  G.add_edges_from((source, retweeter, {'time': time}) for source, retweeter, time in edges)

  return G


def combine_datasets():
  t15 = 'datasets/twitter15/'
  t16 = 'datasets/twitter16/'

  df15 = pd.read_csv(t15 + 'source_tweets.txt', sep = '\t', header = None, names = ['tweet_id', 'tweet'])
  df16 = pd.read_csv(t16 + 'source_tweets.txt', sep='\t', header=None, names=['tweet_id', 'tweet'])
  df   = pd.concat([df15, df16]).drop_duplicates().reset_index(drop=True)
  df.to_csv(PATH + 'source_tweets.txt', sep='\t', index=False, header=False)

  l15 = pd.read_csv(t15 + 'label.txt', sep = ':', header = None, names = ['label', 'tweet_id'])
  l16 = pd.read_csv(t16 + 'label.txt', sep = ':', header = None, names = ['label', 'tweet_id'])
  l = pd.concat([l15, l16]).drop_duplicates().reset_index(drop=True)
  l.to_csv(PATH + 'label.txt', sep=':', index=False, header=False)

  return None


def find_skewed_columns(df, threshold = 2.5):

  skewness = df.skew()

  right_skewed = []
  left_skewed  = []
  for col, skew in skewness.items():
    # right skew
    if skew >= threshold:
      right_skewed.append(col)
    # left skew
    elif skew <= -threshold:
      left_skewed.append(col)

  return right_skewed, left_skewed

def log_transform(df):
  return np.log1p(df)
