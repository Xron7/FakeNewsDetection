from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

PATH = "datasets/twitter1516/"
THRESHOLDS = [0, 5, 30, 60, 180, 480, 720, 1440]
EXCLUDE_COLUMNS = ["poster", "label"]

# for scoring users
WEIGHTS = {"true": 1, "non-rumor": 1, "unverified": 0, "false": -1}
MAX_RT_SCORE = 0.8
ALPHA = 1

RT_UNDER_COLUMNS = []
for t in THRESHOLDS[1:]:
    RT_UNDER_COLUMNS.append(f"rts_under_{t}_min")

MODELS = {
    "logistic": LogisticRegression,
    "forest": RandomForestClassifier,
    "catboost": CatBoostClassifier,
}

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler}
