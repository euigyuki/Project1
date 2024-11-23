import pandas as pd

# Load the CSV file
file_path = "sentences_edited_150-175.csv"
output_file = "extracted_sentences_150-175.txt"

sentences_df = pd.read_csv(file_path)

# Extract the first value (sentence) from each row
sentences = sentences_df["sentence"].tolist()

# Write sentences to a text file
with open(output_file, "w", encoding="utf-8") as f:
    for sentence in sentences:
        f.write(sentence + "\n")

print(f"Sentences have been extracted and saved to {output_file}")
