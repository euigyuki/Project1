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
    verb_csv_path: Path
    kld_csv_path: Path
    output_csv_path: Path
    
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"  


def main():
    #input
    
    captions_filepaths = ["captions1.csv", "captions2.csv"]
    vlm_filepaths = ["images1.csv", "images2.csv"]
    finalized_captions = ["../finalized_captions/finalized_captions.csv"]
    #output
    js_output_csv = "image_to_jensenshannon_divergences.csv"

    evaluator = VLMAnnotationEvaluator(captions_filepaths, vlm_filepaths, finalized_captions)
    jsd_captions = evaluator.analyze_image_annotations()
    output_to_csv(jsd_captions, js_output_csv)


if __name__ == "__main__":
    main()
