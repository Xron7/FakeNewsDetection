import pandas            as pd
import networkx          as nx
import numpy             as np
import joblib

from sklearn.metrics         import accuracy_score, roc_auc_score, log_loss, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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

  print('removed columns:', highly_corr_cols)

  return df.drop(columns=highly_corr_cols)


def get_important_features(X, y, model, n = 15):

  model.fit(X, y)

  importance = pd.Series(model.feature_importances_, index=X.columns)
  top_features = importance.nlargest(n).index.tolist()

  return top_features


def perform_grid_search(pipeline, param_grid, X_train, y_train):
  cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
  grid_search = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, verbose=1)

  grid_search.fit(X_train, y_train)

  print(f"Best Parameters: {grid_search.best_params_}")
  print(f"Best Model Accuracy: {grid_search.best_score_}")

  return grid_search


def evaluate_model(model, X_test, y_test):
  y_pred = model.predict(X_test)
  y_proba = model.predict_proba(X_test)[:, 1]

  print("Accuracy:", accuracy_score(y_test, y_pred))
  print("ROC-AUC:", roc_auc_score(y_test, y_proba))
  print("Log Loss:", log_loss(y_test, y_proba))
  print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
  print("\nClassification Report:\n", classification_report(y_test, y_pred))

  return None


def extract_tweets_file(file_name):
  df = pd.read_csv(PATH + file_name)
  df = df['tweet']
  df.to_csv(PATH + 'tweets.txt', index = False)

  return None


def save_model(model, name):
  joblib.dump(model, 'models/output/' + name + '.pkl')

  return None

def add_sentiment_scores(df, file = 'sentiment_analysis.csv'):
  sent_df = pd.read_csv(PATH + file)

  sent_df['polarity']  = sent_df['positive'] - sent_df['negative']
  sent_df['intensity'] = np.abs(sent_df['positive'] - sent_df['negative'])

  sent_df = sent_df.drop(columns=['tweet', 'positive', 'negative', 'neutral'])

  return pd.concat([df, sent_df], axis=1), sent_df.columns.tolist()
