import json
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from data.results.captions_annotated_by_humans.organize_captions import extract_annotations
from src.helper.helper_functions import majority_vote_from_distribution
from src.helper.helper_functions import normalize_caption
from typing import List

def return_ground_truth_dict(path_config, df_set):
    ground_truth_path = path_config.ground_truth
    ground_truth_sentences = pd.read_csv(ground_truth_path)
    ground_truth_dict = {}
    for _, row in ground_truth_sentences.iterrows():
        sentence = normalize_caption(row['sentence'])
        classification = [row[q] for q in ['q1', 'q2', 'q3', 'q4'] if pd.notna(row[q]) and str(row[q]).strip()]
        classification_str = '/'.join(classification)
        if sentence in df_set:
            ground_truth_dict[sentence] = classification_str
    return ground_truth_dict


def extract_vlm_annotations(path_config) -> List[dict]:
    """
    Extracts annotations from VLM1 and checks for matches in VLM2 to populate additional annotation fields.
    Returns a list of rows (dicts) for DataFrame creation.
    """
    model_pairs = {
        "flux": (path_config.flux1, path_config.flux2),
        "dalle": (path_config.dalle1, path_config.dalle2),
        "midjourney": (path_config.midjourney1, path_config.midjourney2),
    }
    finalized_captions = pd.read_csv(path_config.finalized_captions)
    finalized_to_original_mapping= {
        normalize_caption(row["Finalized Sentence"]): normalize_caption(row["Original Sentence"])
        for _, row in finalized_captions.iterrows()
    }
    

    rows = []
    for model_name, (file1, file2) in model_pairs.items():
        # Load VLM1 data
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        df1["caption"] = df1["caption"].apply(normalize_caption)
        df2["caption"] = df2["caption"].apply(normalize_caption)
        df1_set = set(df1["caption"])
        # Index df2 by caption for quick lookup
        df2_dict = {
            row["caption"]: json.loads(row["probs"]) for _, row in df2.iterrows()
        }
        adjudications={
            row["caption"]: row["adjudicated"] for _, row in df2.iterrows()

        }
        ground_truth_dict = return_ground_truth_dict(path_config,df1_set)

        for _, row in df1.iterrows():
            caption = normalize_caption(row.get("caption", ""))
            url = row.get("url", "")
            prob1 = json.loads(row.get("probs",[]))
            if caption in df2_dict:
                adjudication = adjudications.get(caption, None)
                annotation = majority_vote_from_distribution(df2_dict[caption],adjudication)
            else:
                annotation = majority_vote_from_distribution(prob1)
            if caption in finalized_to_original_mapping:
                original_caption = finalized_to_original_mapping[caption]
                ground_truth_annotation = ground_truth_dict.get(original_caption, "")
            else:
                ground_truth_annotation = ground_truth_dict.get(caption, "")
            rows.append({
                "WorkerId": model_name,
                "Input.sentence": caption,
                "Input.image_url": url,
                "Answer.taskAnswers": annotation,
                "GoldStandardfromChrisWard": ground_truth_annotation  
            })
    df = pd.DataFrame(rows)
    return df



@dataclass
class PathConfig:
    input_csvs: List[Path]
    flux1: Path
    flux2: Path
    dalle1: Path
    dalle2: Path
    midjourney1: Path
    midjourney2: Path
    ground_truth: Path
    finalized_captions: Path
    output_csv: Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT

def main():
    path_config = PathConfig(
        input_csvs=None,
        flux1 = DATA_DIR/"results"/"vlm_annotations"/"flux_judgements_for_ken.csv",
        flux2 = DATA_DIR/"results"/"vlm_annotations"/"flux_judgements_for_ken2.csv",
        dalle1 = DATA_DIR/"results"/"vlm_annotations"/"dalle_judgements_for_ken.csv",
        dalle2 = DATA_DIR/"results"/"vlm_annotations"/"dalle_judgements_for_ken2.csv",
        midjourney1 = DATA_DIR/"results"/"vlm_annotations"/"midjourney_judgements_for_ken.csv",
        midjourney2 = DATA_DIR/"results"/"vlm_annotations"/"midjourney_judgements_for_ken2.csv",
        ground_truth = DATA_DIR/"results"/"chris_ward_sentences"/"chris_ward_ground_truth.csv",
        finalized_captions = DATA_DIR/"finalized_captions"/"finalized_captions.csv",
        output_csv=DATA_DIR/"results"/"vlm_annotations"/"images_annotated_by_vlms.csv"
    )   
  

    extract_annotations(path_config,extract_vlm_annotations, "Input.sentence", entity="vlms")

if __name__ == "__main__":
    main()
