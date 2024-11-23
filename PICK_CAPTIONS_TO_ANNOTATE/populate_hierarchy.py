import pandas as pd
import ast
import yaml


def populate_hierarchy(file_path, pick):
    df = pd.read_csv(file_path)
    print("length of df", len(df))
    with open("data/helper/combinations.yaml", "r") as file:
        all_combinations = yaml.safe_load(file)
    output_rows = []
    unmatched_combinations = []
    for i in all_combinations:
        best_match = None
        max_elementwise_KLD = -float("inf")
        for index, row in df.iterrows():
            max_index_tuple = ast.literal_eval(row["Max_Index"])
            if max_index_tuple == i:
                word = row["Word"]
                print("word", word)
                if word not in i:
                    if row["Max_elementwise_KL"] > max_elementwise_KLD:
                        max_elementwise_KLD = row["Max_elementwise_KL"]
                        best_match = [
                            i,
                            row["Word"],
                            row["KL_Divergence"],
                            row["Max_elementwise_KL"],
                        ]

        # If a match was found, append the best match row for this combination
        if best_match:
            output_rows.append(best_match)
        else:
            # If no match was found, add a placeholder row
            output_rows.append([i, None, None, None])
            unmatched_combinations.append(i)

    # Output debug information for any unmatched combinations
    if unmatched_combinations:
        print(f"No matching rows for these combos: {unmatched_combinations}")

    # Create a new DataFrame with the output rows
    output_df = pd.DataFrame(
        output_rows,
        columns=["Combination", "Word", "KL_Divergence", "Max_elementwise_KL"],
    )
    output_path = f"hierarchy_populated_{pick}.csv"
    output_df.to_csv(output_path, index=False)

    print(f"Hierarchy populated and saved to {output_path}")


def main():
    """input and output file paths"""
    KLD = "../data/kl_divergence"
    """input and output file paths"""

    iterables = ["top_third", "second_third", "third_third"]
    for pick in iterables:
        populate_hierarchy(f"{KLD}kl_divergence_sorted_{pick}.csv", pick)


if __name__ == "__main__":
    main()
