import csv

def count_rows_with_value(csv_file, wordcount_path, threshold=4):
    count = 0
    verb_row_mapping = []  # List to store verbs and their row numbers
    
    # Load original word counts for row references
    with open(wordcount_path, 'r') as wordcount_file:
        original_rows = list(csv.reader(wordcount_file))
    
    # Process the edited CSV
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header if the CSV has one
        row_number = 1  # To match rows with the word_counts_and_combinations.csv
        for row in reader:
            for element in row[1:]:
                if element.strip() and float(element) >= threshold:  # Check for non-empty strings
                    count += 1
                    verb = row[0]
                    verb_row_mapping.append((verb, row_number))
                    break
            row_number += 1  # Increment the row number

    print("Verbs and their corresponding rows:")
    for verb, row in verb_row_mapping:
        print(f"Verb: {verb}, Row: {row}")
    
    return count


def filter_rows_with_value(input_csv, output_csv, threshold=4):
    """
    Filter rows from a CSV file based on a threshold and save the filtered rows into a new CSV file.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the filtered rows into a new CSV file.
        threshold (float): The threshold value to filter rows.
    """
    filtered_rows = []
    
    with open(input_csv, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read and save the header
        filtered_rows.append(header)  # Include the header in the output
        
        for row in reader:
            # Check if any value in the row (excluding the first column) meets the threshold
            if any(float(element.strip()) >= threshold for element in row[1:] if element.strip()):
                filtered_rows.append(row)

    # Write the filtered rows to the new CSV file
    with open(output_csv, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(filtered_rows)
    
    print(f"Filtered rows saved to: {output_csv}")
    print(f"length of filtered rows: {len(filtered_rows)}")


# Example usage
if __name__ == "__main__":
    #input
    file_path = "../data/word_counts_and_combinations/word_counts_and_combinations_edited.csv"
    filtered_path = "../data/word_counts_and_combinations/filtered_rows.csv"
    #output
    file_path = "../data/word_counts_and_combinations/word_counts_and_combinations_edited.csv"

    result = count_rows_with_value(file_path, file_path, threshold=4)
    print(f"Number of rows with at least one value >= 4: {result}")

    filter_rows_with_value(file_path,filtered_path, threshold=4)