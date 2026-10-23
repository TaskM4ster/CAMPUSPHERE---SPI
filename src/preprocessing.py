from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "datasets" / "raw" / "student-mat.csv"

TARGET = "risk_label"
EXCLUDED_ALWAYS = ["G3", TARGET]


def risk_label(g3):
    """Convert the final grade (G3) into the three-level risk label."""
    if g3 < 10:
        return "at_risk"
    elif g3 <= 12:
        return "needs_review"
    return "on_track"


def load_data(path=RAW_PATH):
    """Load the Mathematics dataset and add the risk label."""
    df = pd.read_csv(path, sep=";")
    df[TARGET] = df["G3"].apply(risk_label)
    return df


def get_feature_sets(df):
    """Model A: available after period 1 (no G2). Model B: available after period 2."""
    features_a = [c for c in df.columns if c not in EXCLUDED_ALWAYS + ["G2"]]
    features_b = [c for c in df.columns if c not in EXCLUDED_ALWAYS]
    return features_a, features_b


def build_preprocessor(df, features):
    """One-hot encode text columns; standardize numeric columns."""
    part = df[features]
    cat_cols = part.select_dtypes(exclude="number").columns.tolist()
    num_cols = part.select_dtypes(include="number").columns.tolist()
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])