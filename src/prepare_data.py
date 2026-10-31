
from datasets import load_dataset

DATASET_ID = "cngchis/Support-Ticket-Router-12K-Cleaned"


def main():
    dataset = load_dataset(DATASET_ID)

    for split_name in ["train", "validation", "test"]:
        df = dataset[split_name].to_pandas()

        print(f"\n===== {split_name.upper()} =====")
        print("Number of rows:", len(df))
        print("Missing text:", df["text"].isna().sum())
        print("Missing labels:", df["label"].isna().sum())
        print("Number of categories:", df["label"].nunique())
        print("\nCategory counts:")
        print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
