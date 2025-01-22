import pandas as pd

# Step 1: Read the CSV file
file_path = '../../data/finalized_captions/finalized_captions.csv'
original_captions = "../../data/finalized_captions/original_captions.txt"
finalized_captions = "../../data/finalized_captions/finalized_captions.txt"
df = pd.read_csv(file_path)

# Step 2: Extract the two columns
column1 = df['Original Sentence'] 
column2 = df['Finalized sentence'] 

# Step 3: Write each column to a separate text file
column1.to_csv(original_captions, index=False, header=False)  # Save without index or header
column2.to_csv(finalized_captions, index=False, header=False)  # Save without index or header

print("Files have been created")
