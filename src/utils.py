"""
utils.py

Shared helper functions used across the breast-cancer-mlflow notebook.

Keeping these here keeps the notebook clean and makes the helpers
reusable if I ever want to turn this into a proper pipeline script.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute a standard set of binary classification metrics.

    Parameters
    ----------
    y_true  : array-like — ground truth labels
    y_pred  : array-like — predicted class labels
    y_prob  : array-like, optional — predicted probabilities for the positive
              class. When provided, ROC-AUC is included in the output.

    Returns
    -------
    dict with keys: accuracy, f1, precision, recall, (roc_auc)
    """
    metrics = {
        "accuracy" : accuracy_score(y_true, y_pred),
        "f1"       : f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall"   : recall_score(y_true, y_pred),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_confusion_matrix(y_true, y_pred, run_name, class_names, out_dir="/tmp"):
    """
    Generate and save a confusion matrix heatmap.

    Parameters
    ----------
    y_true      : array-like — ground truth labels
    y_pred      : array-like — predicted labels
    run_name    : str — used in the plot title and filename
    class_names : list[str] — label names for axis ticks
    out_dir     : str — directory to save the PNG

    Returns
    -------
    str — absolute path to the saved PNG file
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {run_name}")
    plt.tight_layout()

    filename = f"cm_{run_name.replace(' ', '_')}.png"
    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath, dpi=100)
    plt.close()
    return filepath


def save_classification_report(y_true, y_pred, class_names, out_dir="/tmp", prefix=""):
    """
    Write a classification report to a .txt file.

    Parameters
    ----------
    y_true      : array-like
    y_pred      : array-like
    class_names : list[str]
    out_dir     : str
    prefix      : str — optional prefix for the filename

    Returns
    -------
    str — absolute path to the saved .txt file
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    filename = f"{prefix}_report.txt" if prefix else "report.txt"
    filepath = os.path.join(out_dir, filename)
    with open(filepath, "w") as f:
        f.write(report)
    return filepath


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(params, random_state=42):
    """
    Build a sklearn Pipeline (StandardScaler + classifier) from a params dict.

    Supported model keys: "LogisticRegression", "RandomForest", "GradientBoosting"

    Parameters
    ----------
    params       : dict — must contain "model" key plus model-specific keys
    random_state : int

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model_name = params["model"]

    if model_name == "LogisticRegression":
        clf = LogisticRegression(
            C=params.get("C", 1.0),
            solver=params.get("solver", "lbfgs"),
            max_iter=params.get("max_iter", 2000),
            random_state=random_state,
        )
    elif model_name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            random_state=random_state,
        )
    elif model_name == "GradientBoosting":
        clf = GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
