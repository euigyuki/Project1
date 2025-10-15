import pandas as pd

def load_first_column(file_path: str) -> list:
    # Read the first column of the CSV file into a DataFrame
    df = pd.read_csv(file_path, usecols=[1], header=None)
    # Convert the DataFrame column to a list
    return df.iloc[:, 0].tolist()