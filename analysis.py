"""
CISC7204 Practice 09 - Diabetes Risk Factors: An Exploratory Data Analysis
==========================================================================

Research question:
    Which health indicators are most strongly associated with the onset of
    diabetes, and how well can a simple logistic regression model separate
    diabetic from non-diabetic patients using them?

Dataset:
    Pima Indians Diabetes Database (UCI Machine Learning Repository).
    Publicly available for research and educational purposes.
    Source: https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes

Tools:
    Python 3, pandas, NumPy, Matplotlib, Seaborn, scikit-learn.

How to run (from this folder):
    pip install -r requirements.txt
    python analysis.py

Outputs:
    - data/diabetes_clean.csv        cleaned dataset (zeros imputed as median)
    - figures/*.png                  visualizations
    - console summary + model metrics

Author: [YOUR NAME]
Date:    2026-10-20
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for servers/CI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "figures")
RAW_PATH = os.path.join(DATA_DIR, "diabetes_raw.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "diabetes_clean.csv")
RANDOM_STATE = 42

COLUMNS = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
    "outcome",
]

# For these variables a value of 0 is physiologically impossible and is
# treated as a missing measurement (UCI documentation).
ZERO_IS_INVALID = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]

OUTCOME_LABELS = {0: "Non-diabetic", 1: "Diabetic"}


# --------------------------------------------------------------------------- #
# Data loading and cleaning
# --------------------------------------------------------------------------- #
def load_and_clean():
    """Load the raw CSV and return a cleaned DataFrame."""
    df = pd.read_csv(RAW_PATH, header=None, names=COLUMNS)
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")

    df_clean = df.copy()
    for col in ZERO_IS_INVALID:
        df_clean[col] = df_clean[col].replace(0, np.nan)

    n_missing = int(df_clean.isna().sum().sum())
    print(f"{n_missing} physiologically invalid zeros were treated as missing.")

    # Impute missing values with the median of the non-missing values.
    df_clean[ZERO_IS_INVALID] = df_clean[ZERO_IS_INVALID].apply(
        lambda col: col.fillna(col.median())
    )

    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned dataset saved to {CLEAN_PATH}\n")
    return df_clean


# --------------------------------------------------------------------------- #
# Visualizations
# --------------------------------------------------------------------------- #
def plot_glucose_by_outcome(df):
    """Glucose distribution split by outcome."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="glucose",
        hue="outcome",
        bins=30,
        kde=True,
        palette=["#4C72B0", "#C44E52"],
        ax=ax,
    )
    ax.set_title("Plasma Glucose Concentration by Diabetes Outcome")
    ax.set_xlabel("Glucose (mg/dL)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_glucose_by_outcome.png"), dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df):
    """Correlation heatmap of all features."""
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_correlation_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_bmi_age_scatter(df):
    """BMI vs age scatter coloured by outcome."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="age",
        y="bmi",
        hue="outcome",
        alpha=0.7,
        palette=["#4C72B0", "#C44E52"],
        ax=ax,
    )
    ax.set_title("BMI vs Age by Diabetes Outcome")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("BMI (kg/m^2)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_bmi_age_scatter.png"), dpi=150)
    plt.close(fig)


def plot_pregnancies_by_outcome(df):
    """Boxplot of pregnancies by outcome."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.boxplot(data=df, x="outcome", y="pregnancies", ax=axes[0])
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels([OUTCOME_LABELS[0], OUTCOME_LABELS[1]])
    axes[0].set_title("Number of Pregnancies by Outcome")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Pregnancies")

    sns.boxplot(data=df, x="outcome", y="age", ax=axes[1])
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels([OUTCOME_LABELS[0], OUTCOME_LABELS[1]])
    axes[1].set_title("Age by Outcome")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Age (years)")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_pregnancies_age_boxplot.png"), dpi=150)
    plt.close(fig)


def plot_feature_importance(coef_series):
    """Horizontal bar chart of logistic regression coefficients."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in coef_series]
    coef_series.plot(kind="barh", color=colors, ax=ax)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_title("Logistic Regression Coefficients (positive = raises risk)")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_feature_importance.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
def train_logistic_model(df):
    """Train a simple logistic regression and print evaluation metrics."""
    X = df.drop(columns=["outcome"])
    y = df["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    coef = pd.Series(model.coef_[0], index=X.columns)
    return model, metrics, coef


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    sns.set_theme(style="whitegrid")
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Load and clean
    df = load_and_clean()

    # 2. Summary statistics
    print("Summary statistics (cleaned data):")
    print(df.describe().round(2).to_string())
    print(f"\nClass balance: {df['outcome'].value_counts().to_dict()}")

    # 3. Visualizations
    plot_glucose_by_outcome(df)
    plot_correlation_heatmap(df)
    plot_bmi_age_scatter(df)
    plot_pregnancies_by_outcome(df)
    print(f"\nSaved 4 EDA figures to {FIG_DIR}")

    # 4. Baseline model
    model, metrics, coef = train_logistic_model(df)
    print("\nLogistic regression (80/20 stratified split):")
    for name, value in metrics.items():
        print(f"  {name:<12}: {value:.4f}")

    plot_feature_importance(coef)
    print(f"Saved feature-importance figure to {FIG_DIR}")

    # 5. Key findings printed for the README / reflection
    print("\n--- Key findings ------------------------------------------")
    print("1. Glucose shows the strongest positive association with the outcome "
          "(r = {:.2f}); diabetic patients have visibly higher glucose.".format(
              df["glucose"].corr(df["outcome"])))
    print("2. BMI and age are the next strongest risk factors "
          "(r = {:.2f} and r = {:.2f}).".format(
              df["bmi"].corr(df["outcome"]), df["age"].corr(df["outcome"])))
    print("3. Mean glucose in diabetics vs non-diabetics: "
          "{:.1f} vs {:.1f} mg/dL.".format(
              df.loc[df["outcome"] == 1, "glucose"].mean(),
              df.loc[df["outcome"] == 0, "glucose"].mean()))
    print("4. Baseline logistic regression achieves "
          "{:.1f}% accuracy and ROC-AUC {:.3f}.".format(
              metrics["Accuracy"] * 100, metrics["ROC-AUC"]))
    print("------------------------------------------------------------------")


if __name__ == "__main__":
    main()
