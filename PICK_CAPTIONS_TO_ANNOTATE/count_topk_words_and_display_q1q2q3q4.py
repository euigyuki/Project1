import pandas as pd
from collections import defaultdict
import csv
import yaml


# Function to map raw combination to valid combination
def map_to_valid_combination(q1, q2, q3, q4):
    if q1 == "outdoors" and q2 == "natural":
        return (q1, q2, q4 if q4 != "nan" else "other_unclear")
    else:
        return (q1, q2, q3 if q3 != "nan" else "other_unclear")


def export_to_csv(output_csv, target_words, category_counts, all_combinations):
    csv_data = []
    all_combinations = [tuple(combo) for combo in all_combinations]

    # Prepare the header row
    header = ["Word"] + [" - ".join(combo) for combo in all_combinations]

    # Prepare the data rows
    for word in target_words:
        row = [word]
        for combination in all_combinations:
            count = category_counts[word].get(combination, 0)
            row.append(count)
        csv_data.append(row)

    # Write the data to a CSV file
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        csv_writer.writerows(csv_data)
    print(f"Data has been exported to {output_csv}")


def verify_counts(word_counts, category_counts, target_words, all_combos):
    all_combos = [tuple(combo) for combo in all_combos]
    # Verification step
    for word in target_words:
        print(f"{word}: {word_counts[word]}")
        print(f"Number of categories: {len(category_counts[word])}")
        print("Categories:")
        total_count = 0
        for combination in all_combos:
            count = category_counts[word].get(combination, 0)
            total_count += count
            print(f"    {combination}: {count}")
        print(f"Total count: {total_count}")
        print(f"Word count: {word_counts[word]}")
        if total_count != word_counts[word]:
            print("Warning: Total category count doesn't match word count!")


def main():
    """input and output file paths"""
    #input
    file_path = "../data/verbs/target_verbs.csv"
    all_combinations_path = "../data/helper/combinations.yaml"
    #output
    verbs_path = "../data/verbs/output_verbs.csv"
    output_csv_dir = "../data/word_counts_and_combinations/"
    output_csv = "word_counts_and_combinations.csv"
    """input and output file paths"""

    with open(all_combinations_path, "r") as file:
        yaml_data = yaml.safe_load(file)
        all_combinations = yaml_data["all_combinations"]

    data_verbs = pd.read_csv(file_path)
    print(data_verbs.head())
    # Replace 'Verb' with the actual column name
    target_words = data_verbs["Verb"].tolist()

    # Initialize dictionaries to store counts
    word_counts = defaultdict(int)
    category_counts = defaultdict(lambda: defaultdict(int))

    # Load the data
    data = pd.read_csv(verbs_path)

    # Iterate through rows in the data
    for _, row in data.iterrows():
        verb = row["Verb"]

        # Get q1, q2, q3, and q4 values as strings
        q1, q2, q3, q4 = row["q1"], row["q2"], row["q3"], row["q4"]

        # Map to valid combination
        valid_combination = map_to_valid_combination(q1, q2, q3, q4)

        # Count occurrences of each target word
        word_counts[verb] += 1
        category_counts[verb][valid_combination] += 1

    verify_counts(word_counts, category_counts, target_words, all_combinations)
    csv_dir = output_csv_dir+output_csv
    export_to_csv(
        csv_dir, target_words, category_counts, all_combinations
    )


if __name__ == "__main__":
    main()
