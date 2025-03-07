import sys
import os
import numpy as np
import pandas as pd

# Import load_config from load_yaml.py
from helper_functions.load_yaml import load_config

def kl_divergence_dict(p_dict, q_dict):
    # Ensure both distributions have the same support
    all_keys = set(p_dict.keys()).union(set(q_dict.keys()))

    # Initialize the dictionary to store KL divergence for each key
    kl_divergence_per_key = {}

    # Small constant to avoid division by zero or log of zero
    epsilon = 1e-10

    # Calculate KL divergence for each key
    for key in all_keys:
        p_value = p_dict.get(key, epsilon)
        q_value = q_dict.get(key, epsilon)

        # Clip values to avoid log(0) or division by zero
        p_value = np.clip(p_value, epsilon, 1)
        q_value = np.clip(q_value, epsilon, 1)

        # Compute the contribution to KL divergence for this key
        kl_divergence_per_key[key] = p_value * np.log10(p_value / q_value)
    return kl_divergence_per_key


def elementwise_kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return (p * np.log10(p / q)).tolist()


def generate_qs(counts_dict):
    """Generate q distribution as a normalized list."""
    total_sum = 0
    q_distribution = {}
    for key, value in counts_dict.items():
        total_sum += value
    for key, value in counts_dict.items():
        q_distribution[key] = value / total_sum
    return q_distribution


def generate_p(df):
    """Generate p distribution as a normalized list from the DataFrame."""
    df = df.copy()  # Avoid modifying the original DataFrame
    df["Total_Count"] = df.iloc[:, 1:].sum(axis=1)
    p_distribution = {}
    for index, row in df.iterrows():
        row_dict = {}
        word = row["Word"]
        for col in df.columns[1:-1]: 
            if row["Total_Count"] > 0:  # Avoid division by zero
                df[col] = df[col].astype(float)     # Ensure the column is of type float64
                df.at[index, col] = row[col] / row["Total_Count"]
                row_dict[col] = row[col] / row["Total_Count"]
        p_distribution[word] = row_dict
    return p_distribution

def write_distribution_to_csv(distribution, output_file,add_last_column=False,sort_by_last_column=False):
   
    # Convert the nested dictionary into a DataFrame
    p_df = pd.DataFrame.from_dict(distribution, orient="index")
    
    # Reset the index to make 'Word' a column
    p_df.reset_index(inplace=True)
    p_df.rename(columns={"index": "Word"}, inplace=True)
    
    if add_last_column:
        p_df["KLD_SUM"] = p_df.iloc[:, 1:].sum(axis=1)
    if sort_by_last_column:
        p_df = p_df.sort_values(by=["KLD_SUM"], ascending=False)

    # Write the DataFrame to a CSV file
    p_df.to_csv(output_file, index=False)


def main():
    """input and output file paths"""
    #input
    file_path = "../../data/word_counts_and_combinations/word_counts_and_combinations.csv"
    CFAC_path = "../../data/helper/counts_for_all_combinations.yaml"
    all_combinations_path = "../../data/helper/combinations.yaml"
    #output
    normalized_word_counts = "../../data/normalized_word_counts/normalized_word_counts.csv"
    kl_divergence_by_row = "../../data/kl_divergence/kl_divergence_by_row.csv"
    """input and output file paths"""

    CFAC = load_config(CFAC_path)
    all_combinations = load_config(all_combinations_path)
    all_combos = []
    for combo in all_combinations['all_combinations']:
            all_combos.append(tuple(combo))
    counts_dict ={}
    
    for location, count in zip(all_combos,CFAC['CFAC']):
        processed_combo = [str(item).split(",") for item in location]
        result = " - ".join([item[0] for item in processed_combo])
        counts_dict[result] = count
    

    df = pd.read_csv(file_path)

    # Generate the p distribution
    df = generate_p(df)
    write_distribution_to_csv(df,normalized_word_counts)

    # Generate the q distribution 
    q_distribution = generate_qs(counts_dict)

    # Calculate the KL divergence for each word and store it in a new column
    kl_dict = {}
    for word, value in df.items():
        p={}
        temp = word
        for x,y in value.items():
            p[x] = y

        kl_divergence_per_key = kl_divergence_dict(p, q_distribution)
        kl_dict[word] = kl_divergence_per_key


    write_distribution_to_csv(kl_dict,kl_divergence_by_row,add_last_column=True,sort_by_last_column=True)

    

if __name__ == "__main__":
    main()
