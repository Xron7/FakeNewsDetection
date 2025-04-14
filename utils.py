import pandas            as pd
import networkx          as nx
import numpy             as np
import torch.nn          as nn
import matplotlib.pyplot as plt
import json
import torch

from sklearn.metrics         import accuracy_score, roc_auc_score, log_loss, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tqdm                    import tqdm

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


# call it to use it
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


def log_transform(df):
  return np.log1p(df)


def remove_corr(df, threshold = 0.9):
  df_numeric = df.select_dtypes(exclude=['object'])

  correlation_matrix = df_numeric.corr()

  upper_triangle = correlation_matrix.where(
      np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
  )

  highly_corr_cols = [
      column for column in upper_triangle.columns
      if any(upper_triangle[column] > threshold)
  ]

  for c in highly_corr_cols:
    print('Removed highly correlated:', c)

  return df.drop(columns=highly_corr_cols)


def get_important_features(X, y, model, n = 15):

  model.fit(X, y)

  importance = pd.Series(model.feature_importances_, index=X.columns)
  top_features = importance.nlargest(n).index.tolist()
  print(f'top {n} features:')
  print(top_features)

  return top_features


def perform_grid_search(pipeline, param_grid, X_train, y_train):
  cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
  grid_search = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, verbose=1)

  grid_search.fit(X_train, y_train)

  print(f"Best Parameters: {grid_search.best_params_}")
  print(f"Best Model Accuracy: {grid_search.best_score_}")

  return grid_search


def evaluate_model(model, X_test, y_test, mode):
  y_pred  = model.predict(X_test)
  y_proba = model.predict_proba(X_test)

  print("Accuracy:", accuracy_score(y_test, y_pred))
  if mode == 'binary':
    print("ROC-AUC:", roc_auc_score(y_test, y_proba[:,1]))

  print("Log Loss:", log_loss(y_test, y_proba))
  print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
  print("\nClassification Report:\n", classification_report(y_test, y_pred))

  return y_pred


# call it to use it
def extract_tweets_file(file_name):
  df = pd.read_csv(PATH + file_name)
  df = df['tweet']
  df.to_csv(PATH + 'tweets.txt', index = False)

  return None


def add_sentiment_scores(df, file = 'sentiment_analysis.csv'):
  sent_df = pd.read_csv(PATH + file)

  sent_df['polarity']  = sent_df['positive'] - sent_df['negative']
  sent_df['intensity'] = np.abs(sent_df['positive'] - sent_df['negative'])

  sent_df = sent_df.drop(columns=['tweet', 'positive', 'negative', 'neutral'])

  return pd.concat([df, sent_df], axis=1), sent_df.columns.tolist()


def get_retweet_stats(tweet_id):
  prop_df = construct_prop_df(tweet_id)
  prop_df = prop_df[1:]  # to exclude the poster

  return prop_df[['retweeter_id', 'time_elapsed']]


def score_users_binary(model, X_test, y_pred, user_stats_file = 'user_stats.csv', max_rt_score = 0.8, alpha = 1):
  flag      = 'tested'
  score_col = 'score'

  print('-------------------------------------------------------------------------------------------------------------')
  print('Scoring users of test set...')

  y_pred = [-1 if val == 0 else val for val in y_pred] # to smooth operations

  user_stats_df = pd.read_csv(PATH + user_stats_file, index_col = 0)

  user_stats_df[score_col] = 0.0
  user_stats_df[flag]      = 0

  for i, (x, y) in tqdm(enumerate(zip(X_test.iterrows(), y_pred))):

    poster   = x[1]['user_id']
    tweet_id = x[1]['tweet_id']

    user_stats_df.loc[poster, flag]       = 1
    user_stats_df.loc[poster, score_col] += y

    # retweeter scoring
    retweet_df = get_retweet_stats(tweet_id)
    for j, rt in retweet_df.iterrows():
      rter = rt['retweeter_id']
      t    = rt['time_elapsed']

      user_stats_df.loc[int(rter), flag] = 1

      value = max_rt_score * np.exp(-alpha * max(0,t) / 60)

      user_stats_df.loc[int(rter), score_col] += y * value # true increases, false decreases

  score_df = user_stats_df[user_stats_df[flag] == 1].drop(columns = 'tested')

  score_df.to_csv(PATH + 'user_scores.csv')

  return None


def parse_config(json_file):
  with open(json_file, "r") as file:
    return json.load(file)


def calculate_feature_loss(x, x_recon):
  # feature_loss = torch.norm(x - x_recon, p=2)
  mse_loss_fn = nn.MSELoss()
  return mse_loss_fn(x_recon, x)


def calculate_structure_loss(h, structure_pairs):
  row, col = structure_pairs
  h_i = h[row]
  h_j = h[col]

  dot_product = torch.sum(h_i * h_j, dim=1)
  return -torch.log(torch.sigmoid(dot_product) + 1e-8).sum()

def plot_loss(losses, type, lambda_, lr, epochs, dims):
  plt.figure(figsize=(8, 5))
  plt.plot(losses, label=f"{type} Loss", linewidth=2)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("GATE Training Loss Curve")
  params_str = f"λ={'d'}, epochs = {epochs}"
  params_str = (
      f"λ = {lambda_}\n"
      f"lr = {lr}\n"
      f"epochs = {epochs}\n"
      f"dims = {dims}\n"
  )
  plt.plot([], [], ' ', label=params_str)
  plt.legend()
  plt.grid(True)

  plt.savefig(f"plots/{type}_{lambda_}_{lr}_{epochs}_{dims}.png", dpi=300)

  return None
