import numpy as np
import pandas as pd
from helper_functions import load_yaml


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
                df.at[index, col] = row[col] / row["Total_Count"]
    return df


def main():
    """input and output file paths"""
    file_path = "word_counts_and_combinations.csv"
    normalized_word_counts = "normalized_word_counts.csv"
    CFAC_path = "../data/helper/counts_for_all_combinations.yaml"
    all_combinations_path = "../data/helper/combinations.yaml"
    """input and output file paths"""

    counts_for_all_combinations = load_yaml(CFAC_path)
    all_combinations = load_yaml(all_combinations_path)

    # Load the DataFrame
    df = pd.read_csv(file_path)

    # Generate the p distribution
    df = generate_p(df)
    df.to_csv(normalized_word_counts, index=False)

    # Generate the q distribution 
    q_distribution = generate_qs(counts_for_all_combinations)
    print("*" * 50, len(q_distribution))

    # Calculate the KL divergence for each word and store it in a new column
    kl_values = []
    max_indices = []
    max_elementwise_kls = []
    for index, row in df.iterrows():
        p = row[1:-1].values  # Skip 'Word', 'Total_Count'
        q = q_distribution  # Use the same q distribution for all words
        kl_value = kl_divergence(p, q)
        kl_values.append(kl_value)

        # Find the index of the maximum value in the p distribution
        max_index = np.argmax(p)
        max_indices.append(all_combinations[max_index])

        # Calculate element-wise KL divergence
        elementwise_kl = elementwise_kl_divergence(p, q)
        max_elementwise_kl = np.max(elementwise_kl)
        max_elementwise_kls.append(max_elementwise_kl)

    # Add the KL divergence to the DataFrame
    df["KL_Divergence"] = kl_values
    df["Max_Index"] = max_indices  # New column for max index in p distribution
    df["Max_elementwise_KL"] = max_elementwise_kls

    # Sort the DataFrame by KL divergence in descending order
    df_sorted = df.sort_values(by="KL_Divergence", ascending=False)

    # Divide the sorted DataFrame into thirds
    n = len(df_sorted)
    third = n // 3

    # Split into three segments and sort each by 'Max_before_KL'
    top_third = df_sorted.iloc[:third].sort_values(
        by="Max_elementwise_KL", ascending=False
    )
    middle_third = df_sorted.iloc[third: 2 * third].sort_values(
        by="Max_elementwise_KL", ascending=False
    )
    bottom_third = df_sorted.iloc[2 * third:].sort_values(
        by="Max_elementwise_KL", ascending=False
    )

    combined_df = pd.concat([top_third, middle_third, bottom_third])

    # Create separator rows
    separator = pd.DataFrame(
        [["*****"] * len(df_sorted.columns)], columns=df_sorted.columns
    )

    # Concatenate the three parts with separators in between
    final_sorted_df = pd.concat(
        [top_third, separator, middle_third, separator, bottom_third]
    )

    # Save the final sorted DataFrame to a CSV file
    output_path_for_kl_with_divider = "kl_divergence_sorted_with_divider.csv"
    output_path_for_kl = "kl_divergence_sorted.csv"
    final_sorted_df[
        ["Word", "KLD", "Max_Index", "Max_elementwise_KL"]
    ].to_csv(output_path_for_kl_with_divider, index=False)
    combined_df[["Word", "KLD", "Max_Index", "Max_elementwise_KL"]].to_csv(
        output_path_for_kl, index=False
    )


if __name__ == "__main__":
    main()
