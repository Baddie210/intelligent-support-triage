
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

DATASET_ID = "cngchis/Support-Ticket-Router-12K-Cleaned"

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "model_comparison.csv"


def main():
    dataset = load_dataset(DATASET_ID)

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()

    X_train = train_df["text"]
    y_train = train_df["label"]
    X_val = val_df["text"]
    y_val = val_df["label"]

    # Model A: your original word-based TF-IDF approach
    word_model = Pipeline([
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
            LogisticRegression(max_iter=1000, random_state=42),
        ),
    ])

    # Model B: word features combined with character features
    combined_features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=50_000,
                sublinear_tf=True,
            ),
        ),
        (
            "character",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=50_000,
                sublinear_tf=True,
            ),
        ),
    ])

    combined_model = Pipeline([
        ("features", combined_features),
        (
            "classifier",
            LogisticRegression(max_iter=1000, random_state=42),
        ),
    ])

    models = {
        "word_tfidf": word_model,
        "word_and_character_tfidf": combined_model,
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        results.append({
            "model": name,
            "validation_accuracy": accuracy,
            "validation_macro_f1": macro_f1,
        })

        print(f"Validation accuracy: {accuracy:.4f}")
        print(f"Validation macro F1: {macro_f1:.4f}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORT_PATH, index=False)

    print("\n===== MODEL COMPARISON =====")
    print(results_df.to_string(index=False))
    print(f"\nSaved comparison to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
