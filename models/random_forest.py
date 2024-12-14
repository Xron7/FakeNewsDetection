import sys
import os

from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from sklearn.ensemble                import RandomForestClassifier
from sklearn.model_selection         import train_test_split, GridSearchCV
from sklearn.metrics                 import accuracy_score, classification_report, roc_auc_score
from sklearn.compose                 import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline                import Pipeline

from config import EXCLUDE_COLUMNS, PATH
from utils import remove_corr, log_transform

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
numerical_cols = X.columns.tolist()
numerical_cols.remove('tweet')

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
    ('model', RandomForestClassifier())
])

########################################################################################################################
# grid search
param_grid = {
    'model__n_estimators':      [350, 450, 550],
    'model__max_depth':         [10, 20, 30, None],
    'model__min_samples_split': [3, 4],
    'model__min_samples_leaf':  [1, 2],
    'model__max_features':      ['log2'],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Model Accuracy: {grid_search.best_score_}")

########################################################################################################################
# pred
y_pred  = grid_search.predict(X_test)
y_proba = grid_search.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
