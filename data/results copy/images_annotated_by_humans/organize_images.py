import pandas as pd
from dataclasses import dataclass
from typing import List
from data.results.captions_annotated_by_humans.organize_captions import extract_annotations
from pathlib import Path
from src.helper.helper_functions import load_combined_df


def load_images_for_humans(path_config):
    input_csvs= path_config.input_csvs
    df = load_combined_df(input_csvs)
    return df

@dataclass
class PathConfig:
    input_csvs: List[Path]
    output_csv: Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT

if __name__ == "__main__":
    path_config = PathConfig(
        input_csvs=[DATA_DIR/"results"/"images_annotated_by_humans"/"images1.csv", 
                    DATA_DIR/"results"/"images_annotated_by_humans"/"images2.csv"],
        output_csv=DATA_DIR /"results"/"images_annotated_by_humans"/ "images_annotated_by_humans.csv"
    )
    caption_or_image = "Input.image_url"

    extract_annotations(path_config,load_images_for_humans,caption_or_image,entity="human")
