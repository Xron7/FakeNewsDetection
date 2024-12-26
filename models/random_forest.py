import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from sklearn.ensemble                import RandomForestClassifier
from sklearn.model_selection         import train_test_split
from sklearn.compose                 import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline                import Pipeline
from sklearn.preprocessing           import FunctionTransformer, StandardScaler


from config import EXCLUDE_COLUMNS, PATH
from utils import remove_corr, log_transform, get_important_features, perform_grid_search, evaluate_model

########################################################################################################################
# custom functions
log_transformer = FunctionTransformer(log_transform, validate = False)

########################################################################################################################
# read df
df = pd.read_csv(PATH + sys.argv[1])

X = df.drop(columns = EXCLUDE_COLUMNS)
y = df['label']

########################################################################################################################
# remove highly correlated
X = remove_corr(X)

########################################################################################################################
# dimensionality reduction
top_features = get_important_features(X.drop(columns = ['tweet']), y, RandomForestClassifier(), n = 5)
top_features.append('tweet')
X = X[top_features]

########################################################################################################################
# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

########################################################################################################################
# pipeline
numerical_cols = X.columns.tolist()
numerical_cols.remove('tweet')

preprocessor = ColumnTransformer(
    transformers=[
        ('text', CountVectorizer(), 'tweet'),
        ('log', log_transformer, numerical_cols),
        ('scaler', StandardScaler(), numerical_cols)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

########################################################################################################################
# grid search
param_grid = {
    'model__n_estimators':      [400, 500, 600],
    'model__max_depth':         [None],
    'model__min_samples_split': [4, 5, 6],
    'model__min_samples_leaf':  [1],
    'model__max_features':      ['log2'],
}

grid_search = perform_grid_search(pipeline, param_grid, X_train, y_train)

########################################################################################################################
# evaluate
evaluate_model(grid_search, X_test, y_test)
