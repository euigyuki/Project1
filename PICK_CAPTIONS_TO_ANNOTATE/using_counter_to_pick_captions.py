import csv
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

# Download the stopwords if not already downloaded
nltk.download("stopwords", quiet=True)


def analyze_comments_filtered(csv_file, number_of_top_words):
    word_counts = Counter()
    stop_words = set(stopwords.words("english"))
    try:
        with open(csv_file, "r", newline="", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # Skip header

            for row in csv_reader:
                if row:
                    # Split the combined column into parts
                    parts = row[0].split("|")
                    if len(parts) >= 3:
                        comment = parts[2].strip()  # Get the comment part
                        words = re.findall(r"\b\w+\b", comment.lower())
                        filtered_words = [
                            word for word in words if word not in stop_words
                        ]
                        word_counts.update(filtered_words)

        print("Word counts in comments (excluding stop words):")
        for word, count in word_counts.most_common(number_of_top_words):
            print(f"{word}: {count}")

        print(f"\nTotal unique words (excluding stop words): {len(word_counts)}")
        print(f"Total words (excluding stop words): {sum(word_counts.values())}")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Example usage
csv_file_path = "results.csv"
number_of_top_words = 300
analyze_comments_filtered(csv_file_path, number_of_top_words)
