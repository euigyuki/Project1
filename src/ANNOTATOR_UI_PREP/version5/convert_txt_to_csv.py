import csv

def txt_to_csv(input_txt_file, output_csv_file):
    """
    Converts a .txt file into a .csv file with a header row.

    Args:
        input_txt_file (str): Path to the input .txt file.
        output_csv_file (str): Path to the output .csv file.
    """
    try:
        # Read the .txt file
        with open(input_txt_file, 'r', encoding='utf-8') as txt_file:
            lines = txt_file.readlines()

        # Open the .csv file for writing
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            
            # Write the header
            writer.writerow(["sentence"])
            
            # Write each line from the .txt file as a row in the .csv file
            for line in lines:
                # Strip any leading/trailing whitespace or newline characters
                writer.writerow([line.strip()])

        print(f"Successfully converted {input_txt_file} to {output_csv_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

input_txt = "finalized_captions/finalized_captions.txt"  # Replace with your .txt file path
output_csv = "finalized_sentences.csv"  # Desired .csv file name
txt_to_csv(input_txt, output_csv)


input_txt = "finalized_captions/original_captions.txt"  # Replace with your .txt file path
output_csv = "original_sentences.csv"  # Desired .csv file name
txt_to_csv(input_txt, output_csv)