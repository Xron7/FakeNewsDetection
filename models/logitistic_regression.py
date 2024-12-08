import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing   import StandardScaler, FunctionTransformer
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline

from config import EXCLUDE_COLUMNS, PATH
from utils  import log_transform, remove_corr

########################################################################################################################
# custom functions
log_transformer = FunctionTransformer(log_transform, validate = False)

########################################################################################################################
# read df
EXCLUDE_COLUMNS.append('tweet')

df = pd.read_csv(PATH + sys.argv[1])

X = df.drop(columns = EXCLUDE_COLUMNS)
y = df['label']

########################################################################################################################
# remove highly correlated
X = remove_corr(X)

########################################################################################################################
## split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

########################################################################################################################
# pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('log_right', log_transformer, X_train.columns),
        ('scaler', StandardScaler(), X_train.columns)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

########################################################################################################################
# fit and pred
pipeline.fit(X_train, y_train)
y_pred       = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

accuracy    = accuracy_score(y_test, y_pred)
loss        = log_loss(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)
report      = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Log Loss:", loss)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# TODO
# Count Vectorizer
# correlation with target
