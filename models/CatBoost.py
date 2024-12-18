import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from catboost                        import CatBoostClassifier
from sklearn.model_selection         import train_test_split
from sklearn.compose                 import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline                import Pipeline
from sklearn.preprocessing           import StandardScaler, FunctionTransformer

from config import EXCLUDE_COLUMNS, PATH
from utils  import perform_grid_search, evaluate_model, log_transform

########################################################################################################################
# custom functions
log_transformer = FunctionTransformer(log_transform, validate = False)

########################################################################################################################
# read df
df = pd.read_csv(PATH + sys.argv[1])

X = df.drop(columns = EXCLUDE_COLUMNS)
y = df['label']

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
        # ('log', log_transformer, numerical_cols),
        # ('scaler', StandardScaler(), numerical_cols)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', CatBoostClassifier(verbose = 0, allow_writing_files = False))
])

########################################################################################################################
# grid search
param_grid = {}

grid_search = perform_grid_search(pipeline, param_grid, X_train, y_train)

########################################################################################################################
# evaluate
evaluate_model(grid_search, X_test, y_test)
