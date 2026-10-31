
from datasets import load_dataset
#imports a ready-made function that downloads and loads datasets,using an existing library

# The name of the dataset we want to use
DATASET_ID = "cngchis/Support-Ticket-Router-12K-Cleaned"
#This identifies the dataset hosted on Hugging Face

#This downloads the dataset if needed and loads it into Python - local cache
def main():
    # Load the dataset from Hugging Face
    dataset = load_dataset(DATASET_ID)

    # Display the available data splits
    print(dataset)

    # Inspect each split
    for split_name, split in dataset.items():
        print(f"\n--- {split_name} ---")
        print("Number of rows:", len(split))
        print("Column names:", split.column_names)

        # Display the first example
        print("\nFirst example:")
        print(split[0])


if __name__ == "__main__":
    main()
