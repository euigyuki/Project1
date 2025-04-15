import pandas as pd

# Function to extract best verbs
def extract_best_verbs(file_paths):
    best_verbs_dict = {}
    verbs_and_locations = {}
    verb_and_bucket_info={}
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        if 'Best Verb' in df.columns:
            for _, row in df.iterrows():
                verb = row['Best Verb']
                location = row['Best Match Column']
                # Update the count of the verb
                best_verbs_dict[verb] = best_verbs_dict.get(verb, 0) + 1
                # Store the best match column for the verb
                if verb not in verbs_and_locations:
                    verbs_and_locations[verb] = location
                if "top" in file_path:
                    verb_and_bucket_info[verb] = "top"
                elif "second" in file_path:
                    verb_and_bucket_info[verb] = "second"
                elif "third" in file_path:
                    verb_and_bucket_info[verb] = "third"
        else:
            print(f"'Best Verb' column not found in {file_path}")
    return best_verbs_dict, verbs_and_locations, verb_and_bucket_info



def cross_check_sentences(best_verbs_set,output_verbs_path, exported_sentences_path):
    # Read the CSV files into DataFrames

    output_verbs = pd.read_csv(output_verbs_path)
    sentences = pd.read_csv(exported_sentences_path)
    sentences['ID'] = range(1, 25341)
    sentences['Image ID'] = (sentences['ID'] - 1) // 5

    merged = output_verbs.merge(sentences, on='ID')
    processed = merged.loc[merged['Processed Sentence'] != 'skip_because_no_change']
    combined_data = processed.loc[processed['Verb'].isin(best_verbs_set)]
    return combined_data

def pick_captions(combined_data,verbs_and_locations,verb_and_bucket_info):
    indices_to_keep = []
    verb_count ={}
    distinct_verbs = {}
    for idx, row in combined_data.iterrows():
        verb = row['Verb']
        location = verbs_and_locations[verb]
        categories = (str(row['q1']), str(row['q2']), str(row['q3']), str(row['q4']))
        combination = ' - '.join(categories).replace(' - nan', '')
        if location==combination:
            indices_to_keep.append(idx)
    combined_data=combined_data.loc[indices_to_keep]
    indices_to_keep = []
    for idx,row in combined_data.iterrows():
        verb = row['Verb']
        verb_count[verb] = verb_count.get(verb, 0) + 1
    for idx,row in combined_data.iterrows():
        verb = row['Verb']
        if verb_count[verb] >= 5:
            indices_to_keep.append(idx)
            distinct_verbs[verb] = distinct_verbs.get(verb, 0) + 1
    combined_data=combined_data.loc[indices_to_keep]
    print(f"Number of distinct verbs: {len(distinct_verbs)}")
    return combined_data

def pick(combined_data):
    unique_verb_image_pairs = combined_data.drop_duplicates(subset=['Verb', 'Image ID'])

    top_captions = unique_verb_image_pairs.groupby('Verb').head(5).reset_index(drop=True)
    return top_captions 


    

# Main script
if __name__ == "__main__":
    #input
    csv_files = [
        '../../data/kl_divergence/hierarchy_populated_top_third.csv',
        '../../data/kl_divergence/hierarchy_populated_second_third.csv',
        '../../data/kl_divergence/hierarchy_populated_third_third.csv'
    ]
    exported_sentences_path = "../../data/exported_sentences/sentences_export25k.csv"
    output_verbs_path = "../../data/verbs/output_verbs.csv"
    #output
    combined_data_path = "../../data/combined_data/combined_data.csv"
    picked_captions_path = "../../data/picked_captions/picked_captions.csv"
    
    best_verbs_dict,verbs_and_locations,verb_and_bucket_info = extract_best_verbs(csv_files)
    best_verbs_set = set(best_verbs_dict.keys())
    
    
    combined_data = cross_check_sentences(best_verbs_set,output_verbs_path, exported_sentences_path)
    

    combined_data.to_csv(combined_data_path, index=False)
    print("Combined data exported to 'combined_data.csv'")

    picked_captions = pick_captions(combined_data,verbs_and_locations,verb_and_bucket_info)
    picked_captions = pick(picked_captions)
    picked_captions.to_csv(picked_captions_path, index=False)