import csv

# Input and output file names
input_file = "sentences.csv"
output_file = "sentences_edited.csv"

# Read the input CSV file and write to the output CSV file
with open(input_file, "r", newline="") as infile, open(
    output_file, "w", newline=""
) as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Write the header (first row) to the output file
    header = next(reader)
    writer.writerow(header)

    # Iterate over the remaining rows
    for i, row in enumerate(reader, start=1):
        # Write the row if it's the first row or if (i+1) mod 5 equals 2
        if i == 1 or (i + 1) % 5 == 2:
            writer.writerow(row)
