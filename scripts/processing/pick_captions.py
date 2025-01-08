import pandas as pd

# Function to extract best verbs
def extract_best_verbs(file_paths):
    best_verbs_dict = {}
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        if 'Best Verb' in df.columns:
            for verb in df['Best Verb']:
                best_verbs_dict[verb] = best_verbs_dict.get(verb, 0) + 1
        else:
            print(f"'Best Verb' column not found in {file_path}")
    return best_verbs_dict

# Function to find sentences containing verbs
def find_sentences_with_verbs(output_verbs_path, verbs_set):
    # Read the CSV file into a DataFrame
    df_sentences = pd.read_csv(output_verbs_path)
    
    # Check if required columns are present
    if 'Sentence' not in df_sentences.columns or 'Verb' not in df_sentences.columns:
        raise ValueError("The required columns ('Sentence', 'Verb') are missing in the sentences file.")
    
    # Filter rows where the 'Verb' column matches any verb in verbs_set
    filtered_sentences = df_sentences[df_sentences['Verb'].isin(verbs_set)]
    temp = len(filtered_sentences)
    return filtered_sentences[['Sentence', 'Verb']]

def cross_check_sentences(sentences_with_verbs, sentences_path, exported_sentences_path):
    # Read the CSV files into DataFrames
    df_sentences = pd.read_csv(sentences_path)
    df_exported_sentences = pd.read_csv(exported_sentences_path)

    matching_sentences = []
    matching_exported_sentences = []

    for _, row in sentences_with_verbs.iterrows():
        sentence = row['Sentence']
        verb = row['Verb']

        matching_rows = df_sentences[df_sentences['sentence'] == sentence]
        matching_rows_exported = df_exported_sentences[df_exported_sentences['Original Sentence'] == sentence]
        # Add the Verb column to the matches for tracking
        if not matching_rows.empty:
            matching_rows = matching_rows.assign(Verb=verb)
            matching_sentences.append(matching_rows)
        if not matching_rows_exported.empty:
            matching_rows_exported = matching_rows_exported.assign(Verb=verb)
            matching_exported_sentences.append(matching_rows_exported)
    print(len(matching_sentences), len(matching_exported_sentences))

    # Combine all matches into DataFrames for further processing
    combined_matching_sentences = pd.concat(matching_sentences, ignore_index=True) if matching_sentences else pd.DataFrame()
    combined_matching_exported = pd.concat(matching_exported_sentences, ignore_index=True) if matching_exported_sentences else pd.DataFrame()
    
    # Merge the two DataFrames on the sentence column
    combined_data = pd.merge(
        combined_matching_sentences,
        combined_matching_exported,
        left_on='sentence',
        right_on='Original Sentence',
        how='inner'
    )
    
    # Display combined data with Verb
    print(combined_data.head())
    print(f"Number of combined rows: {len(combined_data)}")

    return combined_data


# Main script
if __name__ == "__main__":
    #input
    csv_files = [
        '../../data/kl_divergence/hierarchy_populated_top_third.csv',
        '../../data/kl_divergence/hierarchy_populated_second_third.csv',
        '../../data/kl_divergence/hierarchy_populated_third_third.csv'
    ]
    sentences_path = "../../data/sentences/sentences.csv"
    exported_sentences_path = "../../data/exported_sentences/sentences_export25k.csv"
    output_verbs_path = "../../data/verbs/output_verbs.csv"
    #output
    combined_data_path = "../../data/combined_data/combined_data.csv"
    
    best_verbs_dict = extract_best_verbs(csv_files)
    best_verbs_set = set(best_verbs_dict.keys())
    
    sentences_with_verbs = find_sentences_with_verbs(output_verbs_path, best_verbs_set)
    
    combined_data = cross_check_sentences(sentences_with_verbs, sentences_path, exported_sentences_path)
    
    combined_data.to_csv(combined_data_path, index=False)
    print("Combined data exported to 'combined_data.csv'")