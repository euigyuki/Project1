import pandas as pd
from collections import defaultdict
import csv


file_path = "target_verbs.csv"
data_verbs = pd.read_csv(file_path)
print(data_verbs.head())
target_words = data_verbs["Verb"].tolist()  # Replace 'Verb' with the actual column name
print("length of word_list:", len(target_words))
# target_words=["sitting","standing"]

all_combinations = [
    ("indoors", "man-made", "work_education"),
    ("indoors", "man-made", "domestic"),
    ("indoors", "man-made", "recreation"),
    ("indoors", "man-made", "restaurant"),
    ("indoors", "man-made", "transportation_urban"),
    ("indoors", "man-made", "other_unclear"),
    ("outdoors", "man-made", "work_education"),
    ("outdoors", "man-made", "domestic"),
    ("outdoors", "man-made", "recreation"),
    ("outdoors", "man-made", "restaurant"),
    ("outdoors", "man-made", "transportation_urban"),
    ("outdoors", "man-made", "other_unclear"),
    ("outdoors", "natural", "field_forest"),
    ("outdoors", "natural", "body_of_water"),
    ("outdoors", "natural", "mountain"),
    ("outdoors", "natural", "other_unclear"),
]

print("length of all_combinations:", len(all_combinations))


def print_counts(word_counts, category_counts):
    for word in word_counts:
        print(f"{word}: {word_counts[word]}")
        for index, count in category_counts[word].items():
            print(f"Combination {index}: {count}")
        print("length of category_counts[word]:", len(category_counts[word]))


# Function to map raw combination to valid combination
def map_to_valid_combination(q1, q2, q3, q4):
    if q1 == "outdoors" and q2 == "natural":
        return (q1, q2, q4 if q4 != "nan" else "other_unclear")
    else:
        return (q1, q2, q3 if q3 != "nan" else "other_unclear")


def export_to_csv(output_csv, word_counts, category_counts):
    csv_data = []

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


def verify_counts(word_counts, category_counts):
    # Verification step
    for word in target_words:
        print(f"{word}: {word_counts[word]}")
        print(f"Number of categories: {len(category_counts[word])}")
        print("Categories:")
        total_count = 0
        for combination in all_combinations:
            count = category_counts[word].get(combination, 0)
            total_count += count
            print(f"    {combination}: {count}")
        print(f"Total count: {total_count}")
        print(f"Word count: {word_counts[word]}")
        if total_count != word_counts[word]:
            print("Warning: Total category count doesn't match word count!")
        print()


def main():
    print(len(all_combinations))
    # Initialize dictionaries to store counts
    word_counts = defaultdict(int)
    category_counts = defaultdict(lambda: defaultdict(int))

    # Load the data
    data = pd.read_csv("sentences.csv")
    output_csv = "word_counts_and_combinations.csv"

    # Iterate through rows in the data
    for _, row in data.iterrows():
        sentence = row["sentence"].lower()

        # Get q1, q2, q3, and q4 values as strings
        q1, q2, q3, q4 = row["q1"], row["q2"], row["q3"], row["q4"]

        # Map to valid combination
        valid_combination = map_to_valid_combination(q1, q2, q3, q4)

        # Count occurrences of each target word
        for word in target_words:
            if word in sentence:
                word_counts[word] += 1
                category_counts[word][valid_combination] += 1

    verify_counts(word_counts, category_counts)
    export_to_csv(output_csv, word_counts, category_counts)


if __name__ == "__main__":
    main()
