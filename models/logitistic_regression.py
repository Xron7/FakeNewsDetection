import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing   import StandardScaler
# from sklearn.feature_extraction.text import CountVectorizer

from config import EXCLUDE_COLUMNS, PATH

EXCLUDE_COLUMNS.append('tweet')

df = pd.read_csv(PATH + sys.argv[1])

X = df.drop(columns = EXCLUDE_COLUMNS)
y = df['label']

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Count Matrix
# vectorizer     = CountVectorizer()
# X_train_counts = vectorizer.fit_transform(X_train_scaled)
# X_test_counts  = vectorizer.transform(X_test_scaled)

# X_train_scaled = pd.DataFrame(X_train_counts.toarray(), columns=vectorizer.get_feature_names_out())
# X_test_scaled  = pd.DataFrame(X_test_counts.toarray(),  columns=vectorizer.get_feature_names_out())

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred       = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

accuracy    = accuracy_score(y_test, y_pred)
loss        = log_loss(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)
report      = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Log Loss:", loss)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)
