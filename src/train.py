import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from preprocessing import TARGET, build_preprocessor, get_feature_sets, load_data
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
LABELS = ["at_risk", "needs_review", "on_track"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    df = load_data()
    features_a, _ = get_feature_sets(df)
    y = df[TARGET]

    # Same split as the notebook: stratified, fixed seed
    train_idx, test_idx = train_test_split(
        df.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Preprocessing and model are fitted together, on training rows only
    model = make_pipeline(
        build_preprocessor(df, features_a),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(df.loc[train_idx, features_a], y.loc[train_idx])

    y_test = y.loc[test_idx]
    pred = model.predict(df.loc[test_idx, features_a])

    print("Model A test results")
    print(classification_report(y_test, pred, labels=LABELS, digits=3))
    print(confusion_matrix(y_test, pred, labels=LABELS))

    flagged_true = y_test.isin(["at_risk", "needs_review"]).values
    flagged_pred = np.isin(pred, ["at_risk", "needs_review"])
    tp = np.sum(flagged_true & flagged_pred)
    print(f"\nTwo-tier flag: recall={tp / flagged_true.sum():.3f}, "
          f"precision={tp / flagged_pred.sum():.3f}")
    print(f"Students flagged: {flagged_pred.sum()} of {len(flagged_pred)}")

    # Save the pipeline (encoder + model) and the feature list
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model_a_pipeline.joblib")
    joblib.dump(features_a, MODELS_DIR / "model_a_features.joblib")
    print(f"\nSaved model to {MODELS_DIR / 'model_a_pipeline.joblib'}")


if __name__ == "__main__":
    main()