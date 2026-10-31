
from pathlib import Path

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ticket_classifier_combined.joblib"
REPORT_PATH = ROOT / "reports" / "review_thresholds.csv"

DATASET_ID = "cngchis/Support-Ticket-Router-12K-Cleaned"


def main():
    # Load validation data, which we use to make development decisions.
    dataset = load_dataset(DATASET_ID)
    validation_df = dataset["validation"].to_pandas()

    # Load the model trained earlier.
    model = joblib.load(MODEL_PATH)

    texts = validation_df["text"]
    actual = validation_df["label"]

    # Obtain predicted categories and estimated probabilities.
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)

    # Confidence is the probability of the highest-scoring category.
    confidence = probabilities.max(axis=1)

    # True means the model predicted the correct category.
    correct = predictions == actual
    baseline_accuracy = accuracy_score(actual, predictions)

    print("Validation tickets:", len(validation_df))
    print(f"Accuracy without review: {baseline_accuracy:.2%}")
    print("\n===== THRESHOLD COMPARISON =====")

    results = []

    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        # Tickets below the threshold are sent for human review.
        review = confidence < threshold

        total = len(validation_df)
        review_count = int(review.sum())
        automatic_count = total - review_count

        # Errors among tickets that would not be reviewed.
        automatic_errors = int((~correct & ~review).sum())

        # Errors among all tickets, before any human correction.
        total_errors = int((~correct).sum())

        # Of all model errors, what fraction would be flagged?
        errors_caught = int((~correct & review).sum())
        error_capture = (
            errors_caught / total_errors
            if total_errors else 0.0
        )

        automatic_error_rate = (
            automatic_errors / automatic_count
            if automatic_count else 0.0
        )

        row = {
            "threshold": threshold,
            "tickets_for_review": review_count,
            "review_percentage": review_count / total,
            "tickets_not_flagged": automatic_count,
            "errors_among_not_flagged": automatic_errors,
            "error_rate_among_not_flagged": automatic_error_rate,
            "model_errors_caught": errors_caught,
            "fraction_of_model_errors_caught": error_capture,
        }
        results.append(row)

        print(f"\nThreshold: {threshold:.0%}")
        print(f"Tickets flagged for review: {review_count} ({review_count / total:.1%})")
        print(f"Tickets not flagged: {automatic_count}")
        print(f"Errors among tickets not flagged: {automatic_errors}")
        print(f"Fraction of model errors caught: {error_capture:.1%}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(REPORT_PATH, index=False)

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
