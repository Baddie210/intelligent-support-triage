
from datasets import load_dataset
import pandas as pd

DATASET_ID = "cngchis/Support-Ticket-Router-12K-Cleaned"


def main():
    # Load the dataset
    dataset = load_dataset(DATASET_ID)

    # Start with the training split
    df = dataset["train"].to_pandas()

    print("===== DATASET OVERVIEW =====")
    print("Rows and columns:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())

    # Check for missing values
    print("\n===== MISSING VALUES =====")
    print(df.isna().sum())

    # Check for exact duplicate rows
    print("\n===== DUPLICATES =====")
    print("Exact duplicate rows:", df.duplicated().sum())
    print("Duplicate ticket texts:", df["text"].duplicated().sum())

    # Count the number of tickets in each category
    print("\n===== CATEGORY COUNTS =====")
    print(df["label"].value_counts())

    # Show each category as a percentage of the data
    print("\n===== CATEGORY PERCENTAGES =====")
    percentages = df["label"].value_counts(normalize=True) * 100
    print(percentages.round(2))

    # Understand the message lengths
    print("\n===== MESSAGE LENGTHS =====")
    print(df["length"].describe())

    # Display a few examples
    print("\n===== EXAMPLE TICKETS =====")
    print(df[["text", "label", "length"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
