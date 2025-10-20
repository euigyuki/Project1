import pandas as pd
from dataclasses import dataclass
from pathlib import Path


# Define a function to read a CSV file and extract unique best verbs
def extract_best_verbs(path_config) -> dict:
    file_paths = path_config.csv_filepaths
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


@dataclass
class PathConfig:
    csv_filepaths: list[Path]

    
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"  


def main():
    #input
    path_config = PathConfig(
        ## Input paths
        csv_filepaths=[
            DATA_DIR / "kl_divergence" / "hierarchy_populated_top_third.csv",
            DATA_DIR / "kl_divergence" / "hierarchy_populated_second_third.csv",
            DATA_DIR / "kl_divergence" / "hierarchy_populated_third_third.csv"
        ],
       
    )

   # Extract the best verbs and store them in a dictionary
    best_verbs = extract_best_verbs(path_config)

    # Calculate the size of the dictionary
    dictionary_size = len(best_verbs)

    # Print the results
    print("Unique Best Verbs Dictionary:", best_verbs)
    print("Size of the Dictionary:", dictionary_size)


if __name__ == "__main__":
    main()
