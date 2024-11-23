import csv
from collections import Counter
import re
from nltk.corpus import stopwords


def extract_lemmas_from_csv(file_path):
    """
    Reads a CSV file and extracts all unique lemmas into a list.

    :param file_path: Path to the input CSV file.
    :return: List of unique lemmas.
    """
    lemmas = []
    try:
        with open(file_path, mode="r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if "Lemma" in row:  # Ensure the column exists
                    lemmas.append(row["Lemma"])
        # Return unique lemmas
        return list(set(lemmas))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    

def save_verbs_to_csv(verbs, csv):
    """
    Save verbs to a CSV file.
    :param verbs: List of verbs.
    :param csv: Path to the output CSV file.
    """
    with open(csv, "w", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["Verb"])
        for verb in verbs:
            writer.writerow([verb])
        print(f"Verbs saved to {csv}")


def analyze_comments_with_verbs(csv_file, number_of_top_words, verbs):
    """
    Analyze comments from a CSV file and count occurrences of PropBank verbs.
    :param csv_file: Path to the CSV file.
    :param number_of_top_words: Number of top words to display.
    :param verbs: Set of verbs to use as a filter.
    """
    word_counts = Counter()
    stop_words = set(stopwords.words("english"))
    unique_verbs = []

    try:
        with open(csv_file, "r", newline="", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header

            for row in csv_reader:
                if row:
                    # Split the combined column into parts
                    parts = row[0].split("|")
                    if len(parts) >= 3:
                        comment = parts[2].strip()  # Get the comment part
                        words = re.findall(r"\b\w+\b", comment.lower())
                        filtered_verbs = [
                            word
                            for word in words
                            if (word in verbs) and (word not in stop_words)
                        ]
                        # print(filtered_verbs)
                        word_counts.update(filtered_verbs)

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
    verb_path = "../../AMRparsing/output_verbs.csv"  
    csv_file_path = "results.csv"
    target_verbs_csv = "target_verbs.csv"
    """input and output file paths"""

    # Extract lemmas
    lemmas_list = extract_lemmas_from_csv(verb_path)

    # Print the lemmas
    print("Extracted Lemmas:")
    print(lemmas_list)
    print(f"Total unique lemmas: {len(lemmas_list)}")
    number_of_top_words = 300

    # Analyze comments filtered by verbs
    unique_verbs = analyze_comments_with_verbs(
        csv_file_path, number_of_top_words, lemmas_list, target_verbs_csv
    )
    save_verbs_to_csv(unique_verbs, target_verbs_csv)


if __name__ == "__main__":
    main()
