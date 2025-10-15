import pandas as pd

csv_files = ['../../data/kl_divergence/hierarchy_populated_top_third.csv', '../../data/kl_divergence/hierarchy_populated_second_third.csv', '../../data/kl_divergence/hierarchy_populated_third_third.csv']


# Define a function to read a CSV file and extract unique best verbs
def extract_best_verbs(file_paths):
    best_verbs_dict = {}
    
    for file_path in file_paths:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if the "Best Verb" column exists
        if 'Best Verb' in df.columns:
            # Iterate through the Best Verb column and add unique verbs to the dictionary
            for verb in df['Best Verb']:
                best_verbs_dict[verb] = best_verbs_dict.get(verb, 0) + 1
        else:
            print(f"'Best Verb' column not found in {file_path}")
    
    return best_verbs_dict

# List of file paths for the CSV files

# Extract the best verbs and store them in a dictionary
best_verbs = extract_best_verbs(csv_files)

# Calculate the size of the dictionary
dictionary_size = len(best_verbs)

# Print the results
print("Unique Best Verbs Dictionary:", best_verbs)
print("Size of the Dictionary:", dictionary_size)
