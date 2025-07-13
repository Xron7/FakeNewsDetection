"""
Collection of util functions
"""

import pandas as pd
import networkx as nx
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import torch
import seaborn as sns

from sklearn.metrics import (
    auc,
    log_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import plot_importance


from config import PATH, PLOTS_PATH


def construct_prop_df(tweet_id, logging=True):
    propagation_path = PATH + "tree/" + f"{tweet_id}" + ".txt"

    sources = []
    retweeters = []
    times = []
    rt_times = []
    with open(propagation_path, "r") as file:
        # first line for the poster
        first_line = file.readline().strip()
        root, first_retweet = first_line.split("->")

        root = eval(root)
        first_retweet = eval(first_retweet)

        # check if there is time error
        time_shift = 0
        if root[0] != "ROOT":
            if logging:
                print(f"\nDetected time issue for {propagation_path}")
            time_shift = -1 * float(root[2])
            sources.append(root[0])
            retweeters.append(first_retweet[0])
            rt_times.append(
                round(float(first_retweet[2]) - float(root[2]) + time_shift, 2)
            )
            times.append(float(first_retweet[2]) + time_shift)

        for _, line in enumerate(file):
            source, retweet = line.split("->")
            source = eval(source)
            retweet = eval(retweet)

            if source[0] != "ROOT":
                sources.append(source[0])
                retweeters.append(retweet[0])
                rt_times.append(
                    round(float(retweet[2]) - float(source[2]) + time_shift, 2)
                )
                times.append(float(retweet[2]) + time_shift)

        propagation_df = pd.DataFrame(
            {
                "source": sources,
                "retweeter_id": retweeters,
                "time_elapsed": times,
                "time_rt": rt_times,
            }
        )
        return propagation_df


def construct_graph(df):
    G = nx.DiGraph()

    sources = df["source"].tolist()
    retweeters = df["retweeter_id"].tolist()
    times = df["time_elapsed"].tolist()

    edges = zip(sources, retweeters, times)
    G.add_edges_from(
        (source, retweeter, {"time": time}) for source, retweeter, time in edges
    )

    return G


# call it to use it
def combine_datasets():
    t15 = "datasets/twitter15/"
    t16 = "datasets/twitter16/"

    df15 = pd.read_csv(
        t15 + "source_tweets.txt", sep="\t", header=None, names=["tweet_id", "tweet"]
    )
    df16 = pd.read_csv(
        t16 + "source_tweets.txt", sep="\t", header=None, names=["tweet_id", "tweet"]
    )
    df = pd.concat([df15, df16]).drop_duplicates().reset_index(drop=True)
    df.to_csv(PATH + "source_tweets.txt", sep="\t", index=False, header=False)

    l15 = pd.read_csv(
        t15 + "label.txt", sep=":", header=None, names=["label", "tweet_id"]
    )
    l16 = pd.read_csv(
        t16 + "label.txt", sep=":", header=None, names=["label", "tweet_id"]
    )
    l = pd.concat([l15, l16]).drop_duplicates().reset_index(drop=True)
    l.to_csv(PATH + "label.txt", sep=":", index=False, header=False)

    return None


def log_transform(df):
    return np.log1p(df)


def remove_corr(df, threshold=0.9):
    df_numeric = df.select_dtypes(exclude=["object"])

    correlation_matrix = df_numeric.corr()

    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    highly_corr_cols = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]

    for c in highly_corr_cols:
        print("Removed highly correlated:", c)

    return df.drop(columns=highly_corr_cols)


def get_important_features(X, y, model, n=15):
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importance.nlargest(n).index.tolist()
    print(f"top {n} features:")
    print(top_features)

    return top_features


def perform_grid_search(pipeline, param_grid, X_train, y_train):
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, verbose=1)

    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Model Accuracy: {grid_search.best_score_}")

    return grid_search


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("Log Loss:", log_loss(y_test, y_proba))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    labels = ["false", "non-rumor", "true", "unverified"]
    plot_confusion_matrix(y_test, y_pred, labels)
    plot_roc_curve(y_test, y_proba, labels)

    return y_pred


def plot_confusion_matrix(y_test, y_pred, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix(y_test, y_pred, normalize="true"),
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    save_path = f"{PLOTS_PATH}confusion_matrix.png"
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved in '{save_path}'")

    return None


def plot_roc_curve(y_test, y_proba, labels):
    n_classes = len(labels)
    y_test_binarized = label_binarize(y_test, classes=np.arange(n_classes))

    plt.figure(figsize=(8, 6))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{labels[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    save_path = f"{PLOTS_PATH}roc.png"
    plt.savefig(save_path, dpi=300)
    print(f"Roc curves saved in '{save_path}'")

    return None


# call it to use it
def extract_tweets_file(file_name):
    df = pd.read_csv(PATH + file_name)
    df = df["tweet"]
    df.to_csv(PATH + "tweets.txt", index=False)

    return None


def add_sentiment_scores(df, file="sentiment_analysis.csv"):
    sent_df = pd.read_csv(PATH + file)

    sent_df["polarity"] = sent_df["positive"] - sent_df["negative"]
    sent_df["intensity"] = np.abs(sent_df["positive"] - sent_df["negative"])

    sent_df = sent_df.drop(columns=["tweet", "positive", "negative", "neutral"])

    return pd.concat([df, sent_df], axis=1), sent_df.columns.tolist()


def get_retweet_stats(tweet_id):
    prop_df = construct_prop_df(tweet_id)
    prop_df = prop_df[1:]  # to exclude the poster

    return prop_df[["retweeter_id", "time_elapsed"]]


def parse_config(json_file):
    with open(json_file, "r") as file:
        return json.load(file)


def calculate_feature_loss(x, x_recon):
    # feature_loss = torch.norm(x - x_recon, p=2)
    mse_loss_fn = nn.MSELoss()
    return mse_loss_fn(x_recon, x)


def calculate_structure_loss(h, structure_pairs):
    row, col = structure_pairs
    h_i = h[row]
    h_j = h[col]

    dot_product = torch.sum(h_i * h_j, dim=1)
    return -torch.log(torch.sigmoid(dot_product) + 1e-8).sum()


def plot_loss(losses, type_, lambda_, lr, epochs, dims):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label=f"{type_} Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GATE Training Loss Curve")
    params_str = f"λ={'d'}, epochs = {epochs}"
    params_str = f"λ = {lambda_}\nlr = {lr}\nepochs = {epochs}\ndims = {dims}\n"
    plt.plot([], [], " ", label=params_str)
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{PLOTS_PATH}{type_}_{lambda_}_{lr}_{epochs}_{dims}.png", dpi=300)

    return None


def plot_after_cleaning(data, title, metric):
    plt.figure(figsize=(8, 5))
    plt.plot(data, linewidth=2)
    plt.xlabel("# users removed")
    plt.ylabel(f"{metric}")
    plt.title(f"{title}: {metric} vs # users removed")
    plt.grid(True)

    plt.savefig(f"{PLOTS_PATH}{title}_{metric}.png", dpi=300)

    return None
