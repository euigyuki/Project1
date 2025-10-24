import pandas as pd
from dataclasses import dataclass
from typing import List
from src.helper.helper_functions import AnnotationProcessor
from src.helper.helper_functions import load_combined_df,ENTITY_TO_WORKERS
from pathlib import Path

def load_captions_for_humans(path_config):
    input_csvs= path_config.input_csvs
    df = load_combined_df(input_csvs)
    return df


def extract_annotations(path_config,load_func,caption_or_image,entity="human") -> None:
    output_csv = path_config.output_csv
    df = load_func(path_config)
    allowed_workers = ENTITY_TO_WORKERS[entity]
    required_columns = ["WorkerId", caption_or_image, "Answer.taskAnswers"]
    #if entity == "vlms":
    #    required_columns.append("GoldStandardfromChrisWard")
    df_filtered = df[required_columns].copy()
    if entity=="human":
        df_filtered["Answer.taskAnswers"] = df_filtered["Answer.taskAnswers"].apply(
            AnnotationProcessor.process_human_annotation
        )
    df_filtered = df_filtered[df_filtered["WorkerId"].isin(allowed_workers)].copy()
    # Adjust the following column names if they differ in your CSV
    df_filtered = df_filtered.drop_duplicates(subset=["WorkerId", caption_or_image])

     # ✅ Count annotations per person
    counts = df_filtered["WorkerId"].value_counts()
    print(f"✅ Annotations per worker ({entity}):")
    for worker_id, count in counts.items():
        print(f"  {worker_id}: {count} annotations")

    df_filtered.to_csv(output_csv, index=False)
    print(f"\n✅ Saved cleaned annotations to {output_csv}")


@dataclass
class PathConfig:
    input_csvs: List[Path]
    output_csv: Path
    

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT


if __name__ == "__main__":
    path_config = PathConfig(
        input_csvs=[DATA_DIR/"results"/"captions_annotated_by_humans"/"captions1.csv", 
                    DATA_DIR/"results"/"captions_annotated_by_humans"/"captions2.csv"],
        output_csv=DATA_DIR /"results"/"captions_annotated_by_humans"/ "captions_annotated_by_humans.csv"
    )
    extract_annotations(path_config,load_captions_for_humans,"Input.sentence",entity="human")
