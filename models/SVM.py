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
from utils import remove_corr, perform_grid_search, evaluate_model, add_sentiment_scores

########################################################################################################################
# read df
df = pd.read_csv(PATH + sys.argv[1])

########################################################################################################################
# sentiment
df, sent_cols = add_sentiment_scores(df)

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

sent_cols.remove('intensity')
for c in sent_cols:
    numerical_cols.remove(c)

preprocessor = ColumnTransformer(
    transformers=[
        ('text', CountVectorizer(), 'tweet'),
        ('scaler', StandardScaler(), numerical_cols)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', SVC(probability=True, random_state= 42))
])

########################################################################################################################
# grid search
param_grid = {
    'model__C': [0.01, 0.1, 1, 10, 100, 1000],
    'model__gamma': [0.01, 0.05,  0.1],
    'model__kernel': ['rbf', 'linear']
}

grid_search = perform_grid_search(pipeline, param_grid, X_train, y_train)

########################################################################################################################
# evaluate
evaluate_model(grid_search, X_test, y_test)
