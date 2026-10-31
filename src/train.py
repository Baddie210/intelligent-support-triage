
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from datasets import load_dataset
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.pipeline import Pipeline

# Dataset and output locations
DATASET_ID = "cngchis/Support-Ticket-Router-12K-Cleaned"

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def main():
    # 1. Load the existing dataset splits
    dataset = load_dataset(DATASET_ID)

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # 2. Separate messages (inputs) from categories (answers)
    X_train = train_df["text"]
    y_train = train_df["label"]

    X_val = val_df["text"]
    y_val = val_df["label"]

    X_test = test_df["text"]
    y_test = test_df["label"]

    # 3. Create a simple baseline
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)

    # 4. Build the ML pipeline
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=50_000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    # 5. Train the model using training data only
    model.fit(X_train, y_train)

    # 6. Evaluate on validation data for development
    val_predictions = model.predict(X_val)

    print("\n===== VALIDATION RESULTS =====")
    print(
        "Accuracy:",
        round(accuracy_score(y_val, val_predictions), 4),
    )
    print(
        "Macro F1:",
        round(
            f1_score(
                y_val, val_predictions,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
    )

    # 7. Final comparison on the held-out test set
    test_predictions = model.predict(X_test)
    baseline_f1 = f1_score(
        y_test,
        baseline.predict(X_test),
        average="macro",
        zero_division=0,
    )
    model_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro",
        zero_division=0,
    )

    print("\n===== TEST RESULTS =====")
    print("Baseline macro F1:", round(baseline_f1, 4))
    print("ML model macro F1:", round(model_f1, 4))
    print("ML model accuracy:", round(
        accuracy_score(y_test, test_predictions), 4
    ))

    print("\n===== PER-CATEGORY TEST RESULTS =====")
    print(classification_report(
        y_test,
        test_predictions,
        zero_division=0,
    ))

    # 8. Save the confusion matrix
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_predictions,
        xticks_rotation=45,
    )
    display.figure_.tight_layout()
    display.figure_.savefig(
        REPORT_DIR / "confusion_matrix.png",
        dpi=150,
    )
    plt.close(display.figure_)

    # 9. Save examples where the model was wrong
    errors = pd.DataFrame({
        "text": test_df["text"],
        "actual_label": y_test,
        "predicted_label": test_predictions,
    })
    errors = errors[
        errors["actual_label"] != errors["predicted_label"]
    ]
    errors.to_csv(
        REPORT_DIR / "misclassified_tickets.csv",
        index=False,
    )

    # 10. Save the trained pipeline
    joblib.dump(
        model,
        MODEL_DIR / "ticket_classifier.joblib",
    )

    print("\nSaved model and reports.")
    print("Misclassified test examples:", len(errors))


if __name__ == "__main__":
    main()
