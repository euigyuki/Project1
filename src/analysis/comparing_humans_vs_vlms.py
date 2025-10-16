from analysis.displaying_annotations_as_a_probability_distribution import output_to_csv
from analysis.vlm_annotation_evaluator import VLMAnnotationEvaluator
from dataclasses import dataclass
import os
from pathlib import Path

def extract_index_from_filename(filepath):
    # Extract the base name from the file path
    filename = os.path.basename(filepath)
    # Split the filename at the hyphen
    index_str = filename.split('-')[0]
    # Convert the extracted part to an integer
    index = int(index_str)
    return index


@dataclass
class PathConfig:
    captions_filepaths: list[Path]
    vlm_filepaths: list[Path]
    finalized_captions: list[Path]
    js_output_csv: Path
    
    
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"  


def main():
    #input
    path_config = PathConfig(
        ## Input paths
        captions_filepaths=[
            DATA_DIR / "results" / "captions1.csv",
            DATA_DIR / "results" / "captions2.csv"
        ],
        vlm_filepaths=[
            DATA_DIR / "results" / "images1.csv",
            DATA_DIR / "results" / "images2.csv"
        ],
        finalized_captions=[
            DATA_DIR / "finalized_captions" / "finalized_captions.csv"
        ],
        ## Output paths
        js_output_csv=DATA_DIR / "jsd" / "image_to_jensenshannon_divergences.csv"
    )

    evaluator = VLMAnnotationEvaluator(captions_filepaths, vlm_filepaths, finalized_captions)
    jsd_captions = evaluator.analyze_image_annotations()
    output_to_csv(jsd_captions, js_output_csv)


if __name__ == "__main__":
    main()
