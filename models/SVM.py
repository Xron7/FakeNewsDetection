import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from sklearn.svm                     import SVC
from sklearn.model_selection         import train_test_split
from sklearn.compose                 import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline                import Pipeline
from sklearn.preprocessing           import StandardScaler

from config import EXCLUDE_COLUMNS, PATH
from utils import remove_corr, perform_grid_search, evaluate_model

########################################################################################################################
# read df
df = pd.read_csv(PATH + sys.argv[1])

X = df.drop(columns = EXCLUDE_COLUMNS)
y = df['label']

########################################################################################################################
# remove highly correlated
X = remove_corr(X)

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
        ('scaler', StandardScaler(), numerical_cols)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', SVC(kernel="linear", probability=True))
])

########################################################################################################################
# grid search
param_grid = {
    # 'C': [0.1, 1, 10, 100],
    # 'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
    # 'tol': [1e-4, 1e-3, 1e-2]
}

grid_search = perform_grid_search(pipeline, param_grid, X_train, y_train)

########################################################################################################################
# evaluate
evaluate_model(grid_search, X_test, y_test)
