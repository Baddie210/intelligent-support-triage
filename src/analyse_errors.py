
from pathlib import Path

import joblib
import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ticket_classifier.joblib"
REPORT_PATH = ROOT / "reports" / "error_analysis.csv"

DATASET_ID = "cngchis/Support-Ticket-Router-12K-Cleaned"


def main():
    # 1. Load the held-out test dataset
    dataset = load_dataset(DATASET_ID)
    test_df = dataset["test"].to_pandas()

    # 2. Load the model we trained earlier
    model = joblib.load(MODEL_PATH)

    # 3. Predict the category of each test ticket
    predictions = model.predict(test_df["text"])

    # 4. Get the model's estimated probabilities
    probabilities = model.predict_proba(test_df["text"])

    classes = model.named_steps["classifier"].classes_
    results = []

    # 5. Analyse each prediction
    for i in range(len(test_df)):
        scores = probabilities[i]

        # Find the indices of the two highest scores
        top_two = scores.argsort()[-2:][::-1]

        predicted_index = top_two[0]
        second_index = top_two[1]

        results.append({
            "text": test_df.iloc[i]["text"],
            "actual": test_df.iloc[i]["label"],
            "predicted": predictions[i],
            "confidence": scores[predicted_index],
            "second_category": classes[second_index],
            "second_score": scores[second_index],
            "score_gap": (
                scores[predicted_index] - scores[second_index]
            ),
            "correct": (
                test_df.iloc[i]["label"] == predictions[i]
            ),
        })

    # 6. Convert our results into a table
    results_df = pd.DataFrame(results)

    # 7. Save all results for later investigation
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORT_PATH, index=False)

    # 8. Display only the incorrect predictions
    errors = results_df[~results_df["correct"]].sort_values(
        "score_gap"
    )

    print("\n===== ERROR ANALYSIS =====")
    print("Total test tickets:", len(results_df))
    print("Incorrect predictions:", len(errors))

    columns_to_show = [
        "text",
        "actual",
        "predicted",
        "confidence",
        "second_category",
        "second_score",
        "score_gap",
    ]

    print(
        errors[columns_to_show].to_string(
            index=False,
            formatters={
                "confidence": "{:.3f}".format,
                "second_score": "{:.3f}".format,
                "score_gap": "{:.3f}".format,
            },
        )
    )

    print(f"\nFull results saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
