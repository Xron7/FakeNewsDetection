import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from sklearn.neural_network          import MLPClassifier
from sklearn.model_selection         import train_test_split
from sklearn.compose                 import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline                import Pipeline
from sklearn.preprocessing           import FunctionTransformer, StandardScaler

from config import EXCLUDE_COLUMNS, PATH
from utils import remove_corr, log_transform, perform_grid_search, evaluate_model

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
# X = remove_corr(X)

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
    ('model', MLPClassifier(random_state=42))
])

########################################################################################################################
# grid search
param_grid = {
    'model__hidden_layer_sizes': [(100, 50, 25, 10)],
    'model__alpha': [0.0001],
    'model__learning_rate': ['constant'],
    'model__learning_rate_init': [0.001],
    'model__activation': ['relu'],
    'model__max_iter': [1000]
    }

grid_search = perform_grid_search(pipeline, param_grid, X_train, y_train)

########################################################################################################################
# evaluate
evaluate_model(grid_search, X_test, y_test)
