import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline

from preprocessing import TARGET, build_preprocessor, get_feature_sets, load_data

LABELS = ["at_risk", "needs_review", "on_track"]
FLAG = ["at_risk", "needs_review"]
RANDOM_STATE = 42


def make_model(df, features):
    return make_pipeline(
        build_preprocessor(df, features),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def print_two_tier(y_true, pred):
    flagged_true = y_true.isin(FLAG).values
    flagged_pred = np.isin(pred, FLAG)
    tp = np.sum(flagged_true & flagged_pred)
    print(f"Two-tier flag: recall={tp / flagged_true.sum():.3f}, "
          f"precision={tp / flagged_pred.sum():.3f}")
    print(f"Students flagged: {flagged_pred.sum()} of {len(flagged_pred)}")


def main():
    df = load_data()
    features_a, features_b = get_feature_sets(df)
    y = df[TARGET]

    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    # Majority-class baseline (ignores the features)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(train_idx), 1)), y_train)
    base_pred = baseline.predict(np.zeros((len(test_idx), 1)))
    print("BASELINE (test set)")
    print(f"accuracy={accuracy_score(y_test, base_pred):.3f}, "
          f"macro F1={f1_score(y_test, base_pred, average='macro'):.3f}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, feats in [("Model A", features_a), ("Model B", features_b)]:
        print("=" * 60)
        print(name)

        model = make_model(df, feats)
        res = cross_validate(model, df.loc[train_idx, feats], y_train,
                             cv=cv, scoring=["accuracy", "f1_macro"])
        acc, f1 = res["test_accuracy"], res["test_f1_macro"]
        print(f"\n5-fold CV (training set): accuracy={acc.mean():.3f} (±{acc.std():.3f}), "
              f"macro F1={f1.mean():.3f} (±{f1.std():.3f})")

        model.fit(df.loc[train_idx, feats], y_train)
        pred = model.predict(df.loc[test_idx, feats])
        print("\nTest set:")
        print(classification_report(y_test, pred, labels=LABELS, digits=3))
        print(confusion_matrix(y_test, pred, labels=LABELS))
        print_two_tier(y_test, pred)
        print()


if __name__ == "__main__":
    main()