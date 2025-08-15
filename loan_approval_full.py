import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


DATA_FILENAME = "Copy of loan_approval_ - loan_approval_impure.csv.csv"
PLOTS_DIR = "plots"
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    data_path = os.path.join(os.getcwd(), DATA_FILENAME)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    df = pd.read_csv(data_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Target must be present and binary
    df = df[~df["LoanApproved"].isnull()]
    df = df[df["LoanApproved"].isin([0, 1])]

    # Coerce numerics
    for col in ["ApplicantIncome", "LoanAmount", "CreditScore"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing
    df["ApplicantIncome"] = df["ApplicantIncome"].fillna(df["ApplicantIncome"].median())
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["CreditScore"] = df["CreditScore"].fillna(df["CreditScore"].median())

    # Categorical cleanup
    df["SelfEmployed"] = df["SelfEmployed"].fillna("No")
    df["Education"] = df["Education"].fillna("Graduate")

    # Reset index
    df = df.reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IncomePerLoan"] = df["ApplicantIncome"] / (df["LoanAmount"].fillna(0) + 1)
    return df


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def eda_plots(df: pd.DataFrame):
    ensure_plots_dir()

    # Class balance
    plt.figure(figsize=(4, 3))
    sns.countplot(x="LoanApproved", data=df)
    plt.title("Loan Approval Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_balance.png"))
    plt.close()

    # Feature distributions
    df[["ApplicantIncome", "LoanAmount", "CreditScore"]].hist(bins=20, figsize=(9, 3))
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_distributions.png"))
    plt.close()

    # Correlation heatmap (numeric only)
    plt.figure(figsize=(5, 4))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Feature Correlation (numeric only)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"))
    plt.close()

    # Categorical vs target
    plt.figure(figsize=(4, 3))
    sns.countplot(x="Education", hue="LoanApproved", data=df)
    plt.title("Education vs Loan Approval")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "education_vs_target.png"))
    plt.close()

    plt.figure(figsize=(4, 3))
    sns.countplot(x="SelfEmployed", hue="LoanApproved", data=df)
    plt.title("Self Employed vs Loan Approval")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "selfemployed_vs_target.png"))
    plt.close()

    # Boxplots by target
    plt.figure(figsize=(10, 3))
    for i, col in enumerate(["ApplicantIncome", "LoanAmount", "CreditScore"], start=1):
        plt.subplot(1, 3, i)
        sns.boxplot(x="LoanApproved", y=col, data=df)
        plt.title(f"{col} by LoanApproved")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "boxplots_by_target.png"))
    plt.close()


def build_preprocessors():
    numeric = ["ApplicantIncome", "LoanAmount", "CreditScore", "IncomePerLoan"]
    categorical = ["Education", "SelfEmployed"]

    # For LR: scale numeric features
    lr_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )

    # For DT: no scaling
    dt_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )

    return lr_preprocessor, dt_preprocessor


def evaluate_and_print(name: str, y_true: np.ndarray, y_pred: np.ndarray):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    return acc


def plot_roc_curves(y_true, y_prob_lr, y_prob_dt):
    ensure_plots_dir()
    fpr_lr, tpr_lr, _ = roc_curve(y_true, y_prob_lr)
    fpr_dt, tpr_dt, _ = roc_curve(y_true, y_prob_dt)
    auc_lr = roc_auc_score(y_true, y_prob_lr)
    auc_dt = roc_auc_score(y_true, y_prob_dt)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr_lr, tpr_lr, label=f"LogReg AUC={auc_lr:.2f}")
    plt.plot(fpr_dt, tpr_dt, label=f"DecisionTree AUC={auc_dt:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"))
    plt.close()


def quick_shallow_dt_search(X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            preprocessor_dt: ColumnTransformer):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=RANDOM_STATE
    )

    candidates = []
    for crit in ["gini", "entropy"]:
        for depth in [3, 4, 5, 6]:
            for min_leaf in [1, 2]:
                candidates.append(dict(
                    criterion=crit,
                    max_depth=depth,
                    min_samples_split=5,
                    min_samples_leaf=min_leaf,
                    max_features="sqrt",
                    class_weight="balanced",
                    ccp_alpha=0.0,
                ))

    best_model = None
    best_val_acc = -1.0
    best_params = None

    for p in candidates:
        model = Pipeline([
            ("prep", preprocessor_dt),
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE, **p))
        ])
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_val, model.predict(X_val))
        if acc > best_val_acc:
            best_val_acc = acc
            best_model = model
            best_params = p

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print("\nBest shallow DT params:", best_params)
    acc = evaluate_and_print("Decision Tree (shallow tuned)", y_test, y_pred)
    return best_model


def main():
    print("Loading data...")
    df = load_data()

    print("Cleaning data...")
    df = clean_data(df)

    print("Engineering features...")
    df = engineer_features(df)

    print("Running EDA and saving plots in ./plots ...")
    eda_plots(df)

    feature_cols = [
        "ApplicantIncome", "LoanAmount", "CreditScore",
        "IncomePerLoan", "Education", "SelfEmployed"
    ]
    X = df[feature_cols]
    y = df["LoanApproved"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    lr_preprocessor, dt_preprocessor = build_preprocessors()

    # Logistic Regression pipeline
    lr_clf = Pipeline([
        ("preprocess", lr_preprocessor),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")),
    ])

    # Decision Tree baseline pipeline
    dt_clf = Pipeline([
        ("preprocess", dt_preprocessor),
        ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    print("\nFitting Logistic Regression...")
    lr_clf.fit(X_train, y_train)
    y_pred_lr = lr_clf.predict(X_test)
    acc_lr = evaluate_and_print("Logistic Regression", y_test, y_pred_lr)

    print("\nFitting Decision Tree (baseline)...")
    dt_clf.fit(X_train, y_train)
    y_pred_dt = dt_clf.predict(X_test)
    acc_dt = evaluate_and_print("Decision Tree (baseline)", y_test, y_pred_dt)

    # ROC curves
    try:
        y_prob_lr = lr_clf.predict_proba(X_test)[:, 1]
        y_prob_dt = dt_clf.predict_proba(X_test)[:, 1]
        plot_roc_curves(y_test, y_prob_lr, y_prob_dt)
        print("Saved ROC curves to ./plots/roc_curves.png")
    except Exception as e:
        print("Skipping ROC plot:", e)

    # Quick shallow tuning for Decision Tree
    print("\nRunning quick shallow Decision Tree tuning...")
    best_tuned_model = quick_shallow_dt_search(X_train, y_train, X_test, y_test, dt_preprocessor)

    print("\nSummary:")
    print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
    print(f"Decision Tree (baseline) Accuracy: {acc_dt:.4f}")
    print(f"Decision Tree (shallow tuned) Accuracy: {best_tuned_model.score(X_test, y_test):.4f}")
    
    # Save models for the Streamlit app
    print("\nSaving models...")
    import joblib
    
    # Use the best model from the tuning function
    best_tuned_model = quick_shallow_dt_search(X_train, y_train, X_test, y_test, dt_preprocessor)
    
    joblib.dump(lr_clf, 'loan_approval_lr.pkl')
    joblib.dump(best_tuned_model, 'loan_approval_dt.pkl')  # Save the best tuned model
    joblib.dump(lr_preprocessor, 'loan_approval_scaler.pkl')
    print("Models saved: loan_approval_lr.pkl, loan_approval_dt.pkl, loan_approval_scaler.pkl")


if __name__ == "__main__":
    main()

