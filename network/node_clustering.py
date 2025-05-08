import pandas as pd

from config import PATH

df = pd.read_csv(PATH + 'node_embeddings.csv')
n_rows = df.shape[0]

embedding_cols = df.columns.to_list()
embedding_cols.remove('user_id')

# Remove zeros
df = df[embedding_cols].loc[~(df[embedding_cols] == 0.0).all(axis=1)]
print('Removed zero embeddings:')
print(f"Nodes remaining = {df.shape[0]}/{n_rows}")
