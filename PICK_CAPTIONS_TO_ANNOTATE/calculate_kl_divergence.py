import numpy as np
import pandas as pd
import ast


all_combinations = [
    ("indoors", "man-made", "work_education"),
    ("indoors", "man-made", "domestic"),
    ("indoors", "man-made", "recreation"),
    ("indoors", "man-made", "restaurant"),
    ("indoors", "man-made", "transportation_urban"),
    ("indoors", "man-made", "other_unclear"),
    ("outdoors", "man-made", "work_education"),
    ("outdoors", "man-made", "domestic"),
    ("outdoors", "man-made", "recreation"),
    ("outdoors", "man-made", "restaurant"),
    ("outdoors", "man-made", "transportation_urban"),
    ("outdoors", "man-made", "other_unclear"),
    ("outdoors", "natural", "field_forest"),
    ("outdoors", "natural", "body_of_water"),
    ("outdoors", "natural", "mountain"),
    ("outdoors", "natural", "other_unclear"),
]


counts_for_all_combinations = [
    1350,
    1615,
    2030,
    575,
    405,
    610,
    695,
    910,
    4225,
    200,
    6545,
    1185,
    1965,
    2010,
    760,
    260,
]


def kl_divergence(p, q):
    """Calculate the Kullback-Leibler (KL) divergence between two distributions.

    Args:
        p (np.array): True distribution (must sum to 1)
        q (np.array): Approximate distribution (must sum to 1)

    Returns:
        float: KL divergence value
    """
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)

    # To avoid division by zero or log of zero, replace zeros with small epsilon
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
        for col in df.columns[1:-1]:  # Skip 'Word', 'Total_Count', and 'Probability'
            if row["Total_Count"] > 0:  # Avoid division by zero
                df.at[index, col] = row[col] / row["Total_Count"]
    # df['Probability'] = df['Total_Count'] / df['Total_Count'].sum()
    # for index, row in df.iterrows():
    #    print(f"Word: {row['Word']}, Total Count: {row['Total_Count']}, Probability: {row['Probability']}")
    return df


def main():
    # Load the DataFrame
    file_path = "word_counts_and_combinations.csv"
    df = pd.read_csv(file_path)

    # Generate the p distribution
    df = generate_p(df)
    print(df)
    df.to_csv("normalized_word_counts.csv", index=False)

    # Generate the q distribution (assumes the order matches `all_combinations`)
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
    middle_third = df_sorted.iloc[third : 2 * third].sort_values(
        by="Max_elementwise_KL", ascending=False
    )
    bottom_third = df_sorted.iloc[2 * third :].sort_values(
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
        ["Word", "KL_Divergence", "Max_Index", "Max_elementwise_KL"]
    ].to_csv(output_path_for_kl_with_divider, index=False)
    combined_df[["Word", "KL_Divergence", "Max_Index", "Max_elementwise_KL"]].to_csv(
        output_path_for_kl, index=False
    )


if __name__ == "__main__":
    main()
