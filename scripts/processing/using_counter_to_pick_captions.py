import csv
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)


def analyze_comments_filtered(csv_file, number_of_top_words):
    word_counts = Counter()
    stop_words = set(stopwords.words("english"))
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
                        filtered_words = [
                            word for word in words if word not in stop_words
                        ]
                        word_counts.update(filtered_words)

        print("Word counts in comments (excluding stop words):")
        for word, count in word_counts.most_common(number_of_top_words):
            print(f"{word}: {count}")
        tuw = len(word_counts)
        tw = sum(word_counts.values())
        print(f"\nTotal unique words(excluding stop words): {tuw}")
        print(f"Total words(excluding stop words): {tw}")
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def main():
    """input and output file paths"""
    csv_file_path = "results.csv"
    """input and output file paths"""

    number_of_top_words = 300
    analyze_comments_filtered(csv_file_path, number_of_top_words)
