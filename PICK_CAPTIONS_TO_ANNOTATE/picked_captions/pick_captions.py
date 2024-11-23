import csv


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


def filter_rows_by_word(input_file, output_file, target_word):
    """
    Reads the input CSV file in chunks of 5 rows, checks if the target word exists,
    and writes the matching rows to an output CSV file.

    :param input_file: Path to the input CSV file
    :param output_file: Path to the output CSV file
    :param target_word: The word to search for in the rows
    """
    try:
        with open(input_file, "r", encoding="utf-8") as infile, open(
            output_file, "w", encoding="utf-8", newline=""
        ) as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # Write header if present
            header = next(reader, None)
            if header:
                writer.writerow(header)

            # Process rows in chunks of 5
            buffer = []
            for row in reader:
                buffer.append(row)
                if len(buffer) == 5:
                    # Check if target_word exists in any row in the buffer
                    for buf_row in buffer:
                        if any(target_word in cell for cell in buf_row):
                            writer.writerow(buf_row)
                            break
                    buffer = []  # Reset buffer

            # Check the remaining rows in buffer
            for buf_row in buffer:
                if any(target_word in cell for cell in buf_row):
                    writer.writerow(buf_row)

        print(f"Filtered rows containing '{target_word}' written to {output_file}")

    except Exception as e:
        print(f"Error processing the file: {e}")


target_words = extract_lemmas_from_csv("output_verbs.csv")

# Example usage
for word in target_words:
    filter_rows_by_word("results.csv", f"picked_captions_{word}.csv", word)
