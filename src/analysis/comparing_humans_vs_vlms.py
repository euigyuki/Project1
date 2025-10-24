from analysis.vlm_annotation_evaluator import VLMAnnotationEvaluator
from helper.helper_functions import output_to_csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PathConfig:
    captions_filepaths: list[Path]
    vlm_filepaths: list[Path]
    finalized_captions: list[Path]
    judgements_dir: Path
    sanity_checks_dir: Path
    images_js_output_original_captions_csv: Path
    images_js_output_ablated_captions_csv: Path

    
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"  


def main():
    #input
    path_config = PathConfig(
        ## Input paths
        captions_filepaths=[
            DATA_DIR / "results" / "captions_annotated_by_humans" / "captions1.csv",
            DATA_DIR / "results" / "captions_annotated_by_humans" / "captions2.csv"
        ],
        vlm_filepaths=[
            DATA_DIR / "results" / "images_annotated_by_humans" / "images1.csv",
            DATA_DIR / "results" / "images_annotated_by_humans" / "images2.csv"
        ],
        finalized_captions=[
            DATA_DIR / "finalized_captions" / "finalized_captions.csv"],
        judgements_dir=DATA_DIR / "results" / "judgements_for_ken2",
        sanity_checks_dir=DATA_DIR / "results" / "sanity_checks",
        
        ## Output paths
        images_js_output_original_captions_csv=DATA_DIR / "js_divergence" / "image_to_jensenshannon_divergences_original_captions.csv",
        images_js_output_ablated_captions_csv=DATA_DIR / "js_divergence" / "image_to_jensenshannon_divergences_ablated_captions.csv"
    )

    evaluator = VLMAnnotationEvaluator(path_config)
    jensen_shannon_divergences_original, jensen_shannon_divergences_finalized = evaluator.analyze_image_annotations()
    output_to_csv(jensen_shannon_divergences_original, path_config.images_js_output_original_captions_csv)
    output_to_csv(jensen_shannon_divergences_finalized, path_config.images_js_output_ablated_captions_csv)

if __name__ == "__main__":
    main()
