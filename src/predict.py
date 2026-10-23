from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PIPELINE_PATH = MODELS_DIR / "model_a_pipeline.joblib"
FEATURES_PATH = MODELS_DIR / "model_a_features.joblib"

# Two-tier flag: students in either tier are sent for support or review
FLAG_TIERS = ["at_risk", "needs_review"]

# Bands for the probability of needing attention (at_risk + needs_review).
# Cutoffs were chosen from cross-validated calibration on training data only.
BAND_CUTOFFS = (0.3, 0.8)
BAND_NAMES = ("low", "moderate", "high")


def risk_band(flag_probability):
    """Map the probability of needing attention to a band."""
    index = int(np.digitize(flag_probability, BAND_CUTOFFS))
    return BAND_NAMES[index]


def load_model():
    """Load the saved Model A pipeline and its feature list."""
    pipeline = joblib.load(PIPELINE_PATH)
    features = joblib.load(FEATURES_PATH)
    return pipeline, features


def predict_students(students, pipeline=None, features=None):
    """
    Score students and return their risk label, two-tier flag,
    probability band, and the model's probability for each class.

    `students` is a DataFrame with the Model A feature columns.
    The label G3 is not needed and is ignored.
    """
    if pipeline is None or features is None:
        pipeline, features = load_model()

    missing = [c for c in features if c not in students.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = students[features]
    labels = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    prob_df = pd.DataFrame(probs, columns=pipeline.classes_, index=students.index)
    flag_prob = prob_df[FLAG_TIERS].sum(axis=1)

    result = pd.DataFrame(
        {
            "risk_label": labels,
            "flagged": pd.Series(labels).isin(FLAG_TIERS).values,
            "band": [risk_band(p) for p in flag_prob],
            "flag_probability": flag_prob.round(3).values,
        },
        index=students.index,
    )
    return pd.concat([result, prob_df.add_prefix("p_")], axis=1)


if __name__ == "__main__":
    # Smoke test: score the first five students in the dataset
    df = load_data()
    sample = df.head(5)
    output = predict_students(sample)
    output.insert(0, "true_label", sample["risk_label"])
    print(output.round(3).to_string())