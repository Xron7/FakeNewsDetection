import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from sklearn.model_selection         import train_test_split
from sklearn.linear_model            import LogisticRegression
from sklearn.preprocessing           import StandardScaler, FunctionTransformer
from sklearn.compose                 import ColumnTransformer
from sklearn.pipeline                import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

from config import EXCLUDE_COLUMNS, PATH
from utils import log_transform, remove_corr, evaluate_model, save_model, add_sentiment_scores

########################################################################################################################
# custom functions
log_transformer = FunctionTransformer(log_transform, validate = False)

########################################################################################################################
# read df
df = pd.read_csv(PATH + sys.argv[1])

#TODO
df['user_rtXid'] = df['user_rt'] * df['user_id']

########################################################################################################################
# sentiment
df, sent_cols = add_sentiment_scores(df)

########################################################################################################################
# X and y
X = df.drop(columns = EXCLUDE_COLUMNS)
y = df['label']

########################################################################################################################
# remove highly correlated
X = remove_corr(X)
numerical_cols = X.columns.tolist()
numerical_cols.remove('tweet')

sent_cols.remove('intensity')
for c in sent_cols:
    numerical_cols.remove(c)

########################################################################################################################
# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

########################################################################################################################
# pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('text', CountVectorizer(), 'tweet'),
        ('log', log_transformer, numerical_cols),
        ('scaler', StandardScaler(), numerical_cols)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

########################################################################################################################
# fit and evaluate
pipeline.fit(X_train, y_train)
evaluate_model(pipeline, X_test, y_test)

save_model(pipeline, 'log_binary')
