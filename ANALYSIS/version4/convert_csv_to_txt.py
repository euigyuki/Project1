import pandas as pd

input_file = "sentences_edited_first5.csv"
output_file = "sentences_output.txt"

# Read the CSV file
df = pd.read_csv(input_file)

# Assuming the sentences are in a column named 'sentence'
sentences = df["sentence"]

# Write the sentences to a new .txt file
with open(output_file, "w") as file:
    for sentence in sentences:
        file.write(f"{sentence}\n")

print("Sentences have been successfully written to sentences_output.txt")
