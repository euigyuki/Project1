import pandas as pd


def calculate_average(csv_file, caption_or_image):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Check if the column exists in the DataFrame
        if "WorkTimeInSeconds" in df.columns:
            # Calculate the average using the mean() function
            average_time = df["WorkTimeInSeconds"].mean()
            print(
                f"The average work time for {caption_or_image} is {average_time:.2f} seconds."
            )
        else:
            print("Error: 'WorkTimeInSeconds' column not found in the CSV file.")
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{csv_file}' is empty.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


version = 2
image_csv_file_path = f"images{version}.csv"
caption_csv_file_path = f"captions{version}.csv"

calculate_average(caption_csv_file_path, caption_or_image="caption")
calculate_average(image_csv_file_path, caption_or_image="image")
