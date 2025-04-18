from get_fleiss_kappas import load_and_combine_csv
import pandas as pd
from comparing_humans_vs_vlms import change_mturk_annotation_to_more_readable_form
import json
from Project1.scripts.helper.helper_functions import WORKERS



def restructure_annotations(df,sentence_or_image):
    # Load CSV

    # Select relevant columns
    df = df[[sentence_or_image, "WorkerId", "Answer.taskAnswers"]]
    df = df[df["WorkerId"].isin(WORKERS)]


    # Pivot table to have one row per caption, with different annotators' responses as columns
    df_pivot = df.pivot_table(index=sentence_or_image, columns="WorkerId", values="Answer.taskAnswers", aggfunc='first')
    for index, row in df_pivot.iterrows():
        for worker in WORKERS:
            parsed_data = row[worker]
            if pd.notna(parsed_data):
                answer_dict = json.loads(parsed_data)
                df_pivot.at[index, worker] = change_mturk_annotation_to_more_readable_form(answer_dict)
    # Reset index to make captions a column
    df_pivot.reset_index(inplace=True)
    return df_pivot

def organize(filepaths, output_filepath, sentence_or_image):
    combined_df = load_and_combine_csv(filepaths)
    restructured_df = restructure_annotations(combined_df,sentence_or_image)
    restructured_df.to_csv(output_filepath, index=True)  # Saves without the index column


def main():
    #inputs
    captions_filepaths = ["captions1.csv", "captions2.csv"]
    images_filepaths = ["images1.csv", "images2.csv"]
    #outputs
    captions_output_filepath = "total_captions2.csv"
    images_output_filepath = "total_images2.csv"

    organize(captions_filepaths, captions_output_filepath, "Input.sentence")
    organize(images_filepaths, images_output_filepath, "Input.image_url")


if __name__ == "__main__":
    main()
