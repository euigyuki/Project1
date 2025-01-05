import sys
import os
import numpy as np
import pandas as pd

# Add the path to the helper_functions directory
sys.path.append(os.path.abspath("../helper_functions"))

# Import load_config from load_yaml.py
from load_yaml import load_config


def kl_divergence(p, q):
    """Calculate the Kullback-Leibler (KL) divergence between 2 distributions.

    Args:
        p (np.array): True distribution (must sum to 1)
        q (np.array): Approximate distribution (must sum to 1)

    Returns:
        float: KL divergence value
    """
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)

    # To avoid division by zero or log of zero, 
    # replace zeros with small epsilon
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    # Return a scalar value, not an array
    return np.sum(p * np.log10(p / q))


def elementwise_kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return (p * np.log10(p / q)).tolist()


def generate_qs(counts_for_all_combinations):
    """Generate q distribution as a normalized list."""
    total_sum = sum(counts_for_all_combinations)
    return [count / total_sum for count in counts_for_all_combinations]


def generate_p(df):
    """Generate p distribution as a normalized list from the DataFrame."""
    df["Total_Count"] = df.iloc[:, 1:].sum(axis=1)
    for index, row in df.iterrows():
        # Skip 'Word', 'Total_Count', and 'Probability'
        for col in df.columns[1:-1]:  
            if row["Total_Count"] > 0:  # Avoid division by zero
                df[col] = df[col].astype(float)     # Ensure the column is of type float64
                df.at[index, col] = row[col] / row["Total_Count"]

    return df


def main():
    """input and output file paths"""
    #input
    #file_path = "../data/word_counts_and_combinations/word_counts_and_combinations_edited.csv"
    file_path = "../data/word_counts_and_combinations/filtered_rows.csv"
    CFAC_path = "../data/helper/counts_for_all_combinations.yaml"
    all_combinations_path = "../data/helper/combinations.yaml"
    #output
    normalized_word_counts = "../data/normalized_word_counts/normalized_word_counts.csv"
    kl_divergence_by_row = "../data/kl_divergence/kl_divergence_by_row.csv"
    column_maxima_output = "../data/column_maxima/column_maxima.csv"
    """input and output file paths"""

    CFAC = load_config(CFAC_path)
    counts_for_all_combinations = [int(count) for count in CFAC['CFAC']]
    all_combinations = load_config(all_combinations_path)
    all_combos = []
    for combo in all_combinations['all_combinations']:
        all_combos.append(tuple(combo))

    df = pd.read_csv(file_path)

    # Generate the p distribution
    df = generate_p(df)
    df.to_csv(normalized_word_counts, index=False)

    # Generate the q distribution 
    q_distribution = generate_qs(counts_for_all_combinations)

    # Calculate the KL divergence for each word and store it in a new column
    kl_values = []
    kl_matrix = []
    for index, row in df.iterrows():
        p = row[1:-1].values  # Skip 'Word', 'Total_Count'
        q = q_distribution  # Use the same q distribution for all words
        kl_value = kl_divergence(p, q)
        kl_values.append(kl_value)


        # Calculate element-wise KL divergence
        elementwise_kl = elementwise_kl_divergence(p, q)
        kl_matrix.append(elementwise_kl)

    kl_df = pd.DataFrame(kl_matrix, columns=df.columns[1:-1])
    kl_df.insert(0, "Word", df["Word"])
    kl_df.to_csv(kl_divergence_by_row, index=False)
    column_maxima = []
    for col in kl_df.columns[1:]:
        max_kl_value = kl_df[col].max()
        max_word = kl_df.loc[kl_df[col].idxmax(), "Word"]
        column_maxima.append({
            "Column": col,
            "Max_KL_Divergence": max_kl_value,
            "Word": max_word
        })

    column_maxima_df = pd.DataFrame(column_maxima)
    column_maxima_df.to_csv(column_maxima_output, index=False)
 

    

if __name__ == "__main__":
    main()
