import csv
from collections import Counter
    

def save_verbs_to_csv(verbs, output_csv):
    """
    Save verbs to a CSV file.
    :param verbs: List of verbs.
    :param csv: Path to the output CSV file.
    """
    with open(output_csv, "w", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["Verb"])
        for verb in verbs:
            writer.writerow([verb])
        print(f"Verbs saved to {output_csv}")


def analyze_captions_with_verbs(csv_file, number_of_top_words):
    """
    Analyze captions from a CSV file and count occurrences of PropBank verbs.
    :param csv_file: Path to the CSV file.
    :param number_of_top_words: Number of top words to display.
    """
    word_counts = Counter()
    unique_verbs = []

    try:
        with open(csv_file, "r", newline="", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header

            for row in csv_reader:
                if row:
                    word_counts[row[2]] += 1

        print("Verb counts in comments (filtered by PropBank verbs):")
        for word, count in word_counts.most_common(number_of_top_words):
            print(f"{word}: {count}")
            unique_verbs.append(word)

        print(f"\nTotal unique verbs (filtered): {len(word_counts)}")
        print(f"Total verb occurrences(filtered): {sum(word_counts.values())}")
        return unique_verbs
     
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def main():
    """input and output file paths"""
    verb_path = "../data/verbs/output_verbs.csv"
    target_verbs_csv = "../data/verbs/target_verbs.csv"
    """input and output file paths"""
    number_of_top_words = 300

    # Analyze comments filtered by verbs
    unique_verbs = analyze_captions_with_verbs(verb_path, number_of_top_words)
    print(unique_verbs)
    save_verbs_to_csv(unique_verbs, target_verbs_csv)


if __name__ == "__main__":
    main()
