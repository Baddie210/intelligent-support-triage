
# ==========================================
# INTELLIGENT SUPPORT TICKET TRIAGE
# Final Combined Machine Learning Model
# ==========================================

from pathlib import Path

import joblib
import pandas as pd

from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline


# ==========================================
# 1. PROJECT PATHS AND DATASET
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "ticket_classifier_combined.joblib"
REPORT_PATH = REPORT_DIR / "final_test_report.csv"
CONFUSION_PATH = REPORT_DIR / "final_confusion_matrix.csv"

DATASET_ID = "cngchis/Support-Ticket-Router-12K-Cleaned"


# ==========================================
# 2. BUILD THE MACHINE LEARNING MODEL
# ==========================================

def build_model():

    # Word-level TF-IDF
    word_features = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
    )

    # Character-level TF-IDF
    character_features = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
    )

    # Combine both types of text features
    combined_features = FeatureUnion([
        ("word", word_features),
        ("character", character_features),
    ])

    # Create the complete ML pipeline
    model = Pipeline([
        ("features", combined_features),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    return model


# ==========================================
# 3. LOAD THE DATASET
# ==========================================

def main():

    print("\nLoading support ticket dataset...")

    dataset = load_dataset(DATASET_ID)

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    print("Dataset loaded successfully!")

    print("\nDataset sizes:")
    print("Training examples:", len(train_df))
    print("Validation examples:", len(val_df))
    print("Test examples:", len(test_df))


    # ==========================================
    # 4. SEPARATE INPUTS AND CORRECT ANSWERS
    # ==========================================

    X_train = train_df["text"]
    y_train = train_df["label"]

    X_val = val_df["text"]
    y_val = val_df["label"]

    X_test = test_df["text"]
    y_test = test_df["label"]


    # ==========================================
    # 5. CREATE AND TRAIN THE MODEL
    # ==========================================

    print("\nBuilding combined TF-IDF model...")

    model = build_model()

    print("Training model. Please wait...")

    # The model learns from training data only
    model.fit(X_train, y_train)

    print("Training completed!")


    # ==========================================
    # 6. EVALUATE USING VALIDATION DATA
    # ==========================================

    print("\n===== VALIDATION RESULTS =====")

    val_predictions = model.predict(X_val)

    val_accuracy = accuracy_score(
        y_val,
        val_predictions,
    )

    val_macro_f1 = f1_score(
        y_val,
        val_predictions,
        average="macro",
    )

    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation macro F1: {val_macro_f1:.4f}")


    # ==========================================
    # 7. FINAL EVALUATION USING TEST DATA
    # ==========================================

    print("\n===== FINAL TEST RESULTS =====")

    # Evaluate the model on the held-out test set
    test_predictions = model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    test_macro_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro",
    )

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")

    print("\n===== PER-CATEGORY PERFORMANCE =====")

    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )


    # ==========================================
    # 8. SAVE THE EVALUATION REPORTS
    # ==========================================

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save detailed classification metrics
    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORT_PATH)

    # Save the confusion matrix
    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=model.named_steps["classifier"].classes_,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=model.named_steps["classifier"].classes_,
        columns=model.named_steps["classifier"].classes_,
    )

    matrix_df.to_csv(CONFUSION_PATH)


    # ==========================================
    # 9. SAVE THE TRAINED MODEL
    # ==========================================

    joblib.dump(model, MODEL_PATH)

    print("\n===== PROJECT FILES SAVED =====")

    print("Trained model:", MODEL_PATH)
    print("Test report:", REPORT_PATH)
    print("Confusion matrix:", CONFUSION_PATH)

    print("\nFinal model training and evaluation complete!")


# ==========================================
# 10. RUN THE PROGRAM
# ==========================================

if __name__ == "__main__":
    main()
