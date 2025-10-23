from helper.helper_functions import AnnotationProcessor, load_combined_df
import pandas as pd
#from comparing_humans_vs_vlms import change_mturk_annotation_to_more_readable_form
import json
from helper.helper_functions import WORKERS
from pathlib import Path
from dataclasses import dataclass

def restructure_annotations(df,sentence_or_image,sentence_or_image_column="Input.sentence"):
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
                if sentence_or_image_column == "Input.sentence":
                    df_pivot.at[index, worker] = AnnotationProcessor.process_human_annotation(parsed_data)
                else:
                    df_pivot.at[index, worker] = AnnotationProcessor.process_llm_annotation(parsed_data)            
    # Reset index to make captions a column
    df_pivot.reset_index(inplace=True)
    return df_pivot

def organize_captions(path_config, sentence_or_image):
    filepaths = path_config.captions_filepaths
    output_filepath = path_config.captions_output_filepath
    combined_df = load_combined_df(filepaths)
    restructured_df = restructure_annotations(combined_df,sentence_or_image)
    restructured_df.to_csv(output_filepath, index=True)  # Saves without the index column

def organize_images(path_config, sentence_or_image):
    filepaths = path_config.images_filepaths
    output_filepath = path_config.images_output_filepath
    combined_df = load_combined_df(filepaths)
    restructured_df = restructure_annotations(combined_df,sentence_or_image)
    restructured_df.to_csv(output_filepath, index=True)  # Saves without the index column

@dataclass
class PathConfig:
    captions_filepaths: list[Path]
    images_filepaths: list[Path]
    captions_output_filepath: list[Path]
    images_output_filepath: list[Path]
    
    
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"    
    

def main():
    path_config = PathConfig(
        ## Input paths
        captions_filepaths=[
            DATA_DIR / "results" / "captions_annotated_by_humans" / "captions1.csv",
            DATA_DIR / "results" / "captions_annotated_by_humans" / "captions2.csv"
        ],
        images_filepaths=[
            DATA_DIR / "results" / "images_annotated_by_humans" / "images1.csv",
            DATA_DIR / "results" / "images_annotated_by_humans" / "images2.csv"
        ],
    
        captions_output_filepath = DATA_DIR / "results" / "total_captions2.csv",
        images_output_filepath = DATA_DIR / "results" / "total_images2.csv"
    )

   # organize_captions(captions_filepaths, captions_output_filepath, "Input.sentence")
   # organize_images(images_filepaths, images_output_filepath, "Input.image_url")
    organize_captions(path_config, "Input.sentence")
    organize_images(path_config, "Input.image_url")


if __name__ == "__main__":
    main()