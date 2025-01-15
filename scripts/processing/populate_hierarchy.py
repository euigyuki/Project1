import pandas as pd
import ast
import yaml


def populate_hierarchy(file_path):
    # Load data
    df = pd.read_csv(file_path)
   
    # Prepare combinations and results
    
    output_rows = []
    unmatched_combinations = []

    for column_name, column_data in df.iloc[:, :-1].items():
        best_match = None
        best_verb = None
        max_elementwise_KLD = -float("inf")
        best_row_index = None
        if column_data.dtype != 'float64':  # Ensure numeric data
            continue
        # Find the index and value of the maximum element in the column
        current_value = column_data.max()
        current_index = column_data.idxmax()  # Index of the maximum value

        if current_value > max_elementwise_KLD:
            max_elementwise_KLD = current_value
            best_match = column_name
            best_row_index = current_index
            best_verb = df.at[current_index, "Word"]  # Get the corresponding verb from the Word column
            if best_match:
                output_rows.append({
                    "Combination": column_name,
                    "Best Match Column": best_match,
                    "Best Verb": best_verb,
                    "Row Index": best_row_index,
                    "Max KLD": max_elementwise_KLD
                })
            else:
                unmatched_combinations.append(column_name)
  

    # Convert results to DataFrame
    df = pd.DataFrame(output_rows)

    # Log unmatched combinations for debugging
    if unmatched_combinations:
        print(f"Unmatched combinations: {unmatched_combinations}")
    output_df = df.sort_values(by='Max KLD', ascending=False)
    return output_df



def write_output(output_df, output_path, pick):    
    output_df.to_csv(output_path, index=False)

    print(f"Hierarchy populated and saved to {output_path}")


def main():
    """input and output file paths"""
    KLD = "../../data/kl_divergence/"
    #input
    input_path = f"{KLD}kl_divergence_by_row"
    #output
    output_path = f"{KLD}hierarchy_populated"
    """input and output file paths"""

    iterables = ["top_third", "second_third", "third_third"]
    for pick in iterables:
        input = input_path+f"_{pick}.csv"
        output_df = populate_hierarchy(input)
        output = output_path+f"_{pick}.csv"
        write_output(output_df, output, pick)


if __name__ == "__main__":
    main()
