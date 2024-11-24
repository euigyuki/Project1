import csv


def get_sentences(input_file, start_row, end_row, output_file):
    # Read the specified range of rows and write them to a new file
    with open(input_file, "r", newline="", encoding="utf-8") as infile, open(
        output_file, "w", newline="", encoding="utf-8"
    ) as outfile:
        csv_reader = csv.reader(infile)
        csv_writer = csv.writer(outfile)

        # Write the header row
        header = next(csv_reader)
        csv_writer.writerow(header)

        # Skip rows until the start_row
        for _ in range(start_row - 1):
            next(csv_reader, None)

        # Write rows from start_row to end_row (inclusive)
        for i, row in enumerate(csv_reader, start=start_row):
            if i <= end_row:
                csv_writer.writerow(row)
            else:
                break

    print(f"Rows {start_row} to {end_row} have been saved to {output_file}")


def main():
    # Specify the starting and ending row numbers (inclusive)
    start_row = 1  # Change this to your desired starting row
    end_row = 50  # Change this to your desired ending row  

    """input and output file paths"""
    input_file = "../data/sentences/sentences_edited.csv"
    output_file = f"../data/sentences/sentences_edited_first{end_row}.csv"
    """input and output file paths"""

    get_sentences(input_file, start_row, end_row, output_file)


if __name__ == "__main__":
    main()