
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ticket_classifier_combined.joblib"

# Provisional threshold selected from validation experiments.
REVIEW_THRESHOLD = 0.70


def predict_ticket(model, text):
    probabilities = model.predict_proba([text])[0]
    classes = model.named_steps["classifier"].classes_

    # Sort categories from highest to lowest estimated probability.
    ranked_indices = probabilities.argsort()[::-1]

    best_index = ranked_indices[0]
    second_index = ranked_indices[1]

    category = str(classes[best_index])
    confidence = float(probabilities[best_index])

    second_category = str(classes[second_index])
    second_confidence = float(probabilities[second_index])

    needs_review = confidence < REVIEW_THRESHOLD

    return {
        "category": category,
        "confidence": confidence,
        "second_category": second_category,
        "second_confidence": second_confidence,
        "needs_review": needs_review,
    }


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run src/train.py first.")
        return

    model = joblib.load(MODEL_PATH)

    print("=== Intelligent Support Ticket Triage ===")
    print("Type 'exit' to quit.")

    while True:
        text = input("\nEnter a ticket: ").strip()

        if text.lower() == "exit":
            break

        if not text:
            print("Please enter a ticket message.")
            continue

        result = predict_ticket(model, text)

        print("\nPredictions")
        print(
            f"Top category: {result['category']} "
            f"({result['confidence']:.1%})"
        )
        print(
            f"Second category: {result['second_category']} "
            f"({result['second_confidence']:.1%})"
        )

        if result["needs_review"]:
            print("Decision: HUMAN REVIEW RECOMMENDED")
        else:
            print("Decision: CATEGORY SUGGESTION")
            print("A human can still verify the prediction.")


if __name__ == "__main__":
    main()
