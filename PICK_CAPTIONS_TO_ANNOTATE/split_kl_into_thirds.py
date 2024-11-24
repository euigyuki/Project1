import pandas as pd
import math


def main():
    """input and output file paths"""
    input_file = "../data/kl_divergence/kl_divergence_sorted.csv"
    first = "../data/kl_divergence/kl_divergence_sorted_top_third.csv"
    second = "../data/kl_divergence/kl_divergence_sorted_second_third.csv"
    third = "../data/kl_divergence/kl_divergence_sorted_third_third.csv"
    """input and output file paths"""

    # Load the data from the CSV file
    data = pd.read_csv(input_file)

    # Calculate the number of rows for each split
    total_rows = len(data) - 1
    split_size = math.ceil(total_rows / 3)

    # Split the data into thirds
    split_1 = data.iloc[:split_size]
    split_2 = data.iloc[split_size: split_size * 2]
    split_3 = data.iloc[split_size * 2:]

    # Save each split into a separate CSV file
    split_1.to_csv(first, index=False)
    split_2.to_csv(second, index=False)
    split_3.to_csv(third)

    print("Data has been split into thirds ")


if __name__ == "__main__":
    main()
