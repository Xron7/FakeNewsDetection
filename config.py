from sklearn.linear_model import LogisticRegression
from sklearn.ensemble     import RandomForestClassifier
from catboost             import CatBoostClassifier

PATH            = '../datasets/twitter1516/'
THRESHOLDS      = [0, 5, 30, 60, 180, 480, 720, 1440]
EXCLUDE_COLUMNS = ['poster', 'label']

RT_UNDER_COLUMNS = []
for t in THRESHOLDS[1:]:
  RT_UNDER_COLUMNS.append(f'rts_under_{t}_min')

MODELS = {
  'logistic': LogisticRegression,
  'forest':   RandomForestClassifier,
  'catboost': CatBoostClassifier
}
