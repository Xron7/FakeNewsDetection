import sys

import pandas as pd

from sklearn.model_selection         import train_test_split
from sklearn.preprocessing           import FunctionTransformer
from sklearn.compose                 import ColumnTransformer
from sklearn.pipeline                import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

from config import EXCLUDE_COLUMNS, PATH, MODELS, SCALERS
from utils  import log_transform, remove_corr, evaluate_model, add_sentiment_scores, score_users_binary, parse_config, \
    get_important_features, perform_grid_search

########################################################################################################################
# Setup
config = parse_config(sys.argv[1])

dataset = config['dataset']
print(f'dataset = {dataset}')
df = pd.read_csv(PATH + dataset)

model_name = config['model']
model      = MODELS[model_name]
print(f'model = {model_name}')
print('---------------------------------------------------------------------------------------------------------------')

params = config['params']
########################################################################################################################
# Transformations
# log
if config.get('log', False):
    print('Applying log transform')
    log_transformer = FunctionTransformer(log_transform, validate=False)

# sentiment
print('Enhancing with sentiment')
df, sent_cols = add_sentiment_scores(df)

# combinations
for comb in config.get('combinations', []):
    op        = comb['op']
    comb_name = comb['name']

    if op == 'mult':
        df[comb_name] = df[comb['c1']] * df[comb['c2']]

    elif op == 'div':
        df[comb_name] = df[comb['c1']] / (df[comb['c2']] + 0.0000000001)

    print(f'Created combination: {comb_name}')

########################################################################################################################
# X and y
X = df.drop(columns = EXCLUDE_COLUMNS)
y = df['label']

########################################################################################################################
# correlation
if config.get('remove_corr', False):
    X = remove_corr(X)

# dimensionality reduction
top_n = config.get('top_features', 0)
if top_n:
    top_features = get_important_features(X.drop(columns = ['tweet']), y, model(), n = top_n)
    top_features.append('tweet')
    X = X[top_features]

########################################################################################################################
# numerical columns
numerical_cols = X.columns.tolist()
numerical_cols.remove('tweet')

for c in config.get('num_exclude', []):
    numerical_cols.remove(c)
    print(f'Removed {c} from numerical columns list')

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

########################################################################################################################
# pipeline
scaler_name = config.get('scaler', None)
scaler      = SCALERS.get(scaler_name, None)
print(f'scaler = {scaler_name}')

preprocessor = ColumnTransformer(
    transformers=[
        ('text',   CountVectorizer(), 'tweet')        if config.get('count_matrix', False) else None,
        ('log',    log_transformer,   numerical_cols) if config.get('log', False)          else None,
        ('scaler', scaler(),          numerical_cols) if scaler_name                       else None
    ]
)
preprocessor.transformers = [t for t in preprocessor.transformers if t is not None]

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model() if config.get('grid_search', False) else model(**config['params']))
])

print('---------------------------------------------------------------------------------------------------------------')

########################################################################################################################
# fit and evaluate
if config.get('grid_search', False):
    pipeline = perform_grid_search(pipeline, params, X_train, y_train)
else:
    pipeline.fit(X_train, y_train)

y_pred = evaluate_model(pipeline, X_test, y_test, mode = config['mode'])

########################################################################################################################
# scoring
if config.get('score_users', False):
    score_users_binary(pipeline, X_test, y_pred)
