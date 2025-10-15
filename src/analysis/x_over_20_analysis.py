import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from typing import List
from scripts.helper.helper_functions import normalize_caption, WORKERS  
from scripts.helper.helper_functions import LLMS,VLMS
from pathlib import Path

EXPECTED_ANNOTATIONS = 10




class AnnotationProcessor:
    def __init__(self, path_config):
        self.path_config = path_config

    def process(self, loader_func, agents, output_func, grouping_func, classification_transform_func, total_per_sentence):
        df_summary, missing = generate_annotation_report(
            self.path_config,
            loader_func,
            agents,
            grouping_func,
            classification_transform_func,
            total_per_sentence
        )
        save_summary(self.path_config, df_summary, missing, output_func)


def keep_full_classification(label: str) -> str:
    return label

def extract_top_level(label: str) -> str:
    return label.split("/")[0] if label else ""

def extract_mid_level(label: str) -> str:
    return label.split("/")[1] if label else ""


def compute_worker_missing(group, workers, verb):
    """
    Returns list of missing annotation info for a group (one verb).
    """
    per_worker_count = group['WorkerId'].value_counts()
    missing_annotations = []
    for worker in workers:
        count = per_worker_count.get(worker, 0)
        if count < EXPECTED_ANNOTATIONS:
            missing_annotations.append({
                "WorkerId": worker,
                "Verb": verb,
                "Provided": count,
                "Missing": EXPECTED_ANNOTATIONS - count
            })
    return missing_annotations, per_worker_count

def compute_correct_counts(group, mapper, classification_transform_func):
    original_count, processed_count = 0, 0
    for _, row in group.iterrows():
        sentence = normalize_caption(row['Input.sentence'])
        annotation = row['Answer.taskAnswers']
        ground_truth = mapper.get_classification(sentence)
        if ground_truth:
            ground_truth = classification_transform_func(ground_truth)
            annotation = classification_transform_func(annotation)
        if ground_truth == annotation:
            if sentence in mapper.sentences_to_classification:
                original_count += 1
            elif sentence in mapper.processed_sentence_to_classification:
                processed_count += 1
    return original_count, processed_count

def generate_annotation_report(path_config, load_annotations_func, WORKERS_or_LLMS, grouping_func, classification_transform_func, total_per_sentence):
    caption_filepath = load_annotations_func(path_config)
    finalized_captions_filepath = path_config.finalized_captions_filepath

    mapper = ClassificationMapper(finalized_captions_filepath)
    combination = pd.read_csv(caption_filepath)
    selected_columns = filter_valid_annotations(combination, WORKERS_or_LLMS)
    selected_columns = selected_columns.drop_duplicates(subset=["WorkerId", "Input.sentence"])
    grouped, selected_columns = grouping_func(selected_columns, mapper)

    summary_rows = []
    all_missing = []

    for verb, group in grouped:
        row, missing = summarize_group(
            verb, group, WORKERS_or_LLMS, mapper, classification_transform_func, total_per_sentence
        )
        summary_rows.append(row)
        all_missing.extend(missing)

    df_summary = pd.DataFrame(summary_rows)
    return df_summary, all_missing


def filter_valid_annotations(df, workers):
    df = df[df['WorkerId'].isin(workers)]
    return df[['WorkerId', 'Input.sentence', 'Answer.taskAnswers']]


def group_by_verb(df, mapper):
    df = df.copy()
    df['verbs'] = df['Input.sentence'].apply(lambda x: mapper.get_verb(x))
    df = df[df['verbs'] != '']
    grouped = df.groupby('verbs')
    return grouped, df

def summarize_group(verb, group, workers, mapper, classification_transform_func, total_per_sentence):
    missing, counts = compute_worker_missing(group, workers, verb)
    original_correct, processed_correct = compute_correct_counts(group, mapper,classification_transform_func)
    original_pct = original_correct / total_per_sentence
    processed_pct = processed_correct / total_per_sentence
    change = (original_correct-processed_correct)/total_per_sentence
    row = {
        "Verb": verb,
        "Total annotations": len(group),
        "Original correct": original_correct,
        "Processed correct": processed_correct,
        "drop in percentage": change,
        "Original percentage": original_pct,
        "Processed percentage": processed_pct,
    }

    for worker in workers:
        row[f"Worker {worker}"] = f"{counts.get(worker, 0)} / {EXPECTED_ANNOTATIONS}"
    return row, missing



def load_human_annotations(path_config):
    return path_config.captions_filepaths

def load_llm_annotations(path_config):
    return path_config.llm_captions_filepath

def load_vlm_annotations(path_config):
    return path_config.vlm_captions_filepath

def load_human_output_path(path_config):
    return path_config.lvl3_output_filepath_for_humans
def load_llm_output_path(path_config):
    return path_config.lvl3_output_filepath_for_llms
def load_vlm_output_path(path_config):
    return path_config.lvl3_output_filepath_for_vlms

def load_top_level_human_output_path(path_config):
    return path_config.top_level_filepath_for_humans
def load_top_level_llm_output_path(path_config):
    return path_config.top_level_filepath_for_llms
def load_top_level_vlm_output_path(path_config):
    return path_config.top_level_filepath_for_vlms

def load_mid_level_human_output_path(path_config):
    return path_config.mid_level_filepath_for_humans
def load_mid_level_llm_output_path(path_config):
    return path_config.mid_level_filepath_for_llms
def load_mid_level_vlm_output_path(path_config):
    return path_config.mid_level_filepath_for_vlms



def save_summary(path_config, df_summary,missing_annotations,load_func):
    output_filepath = load_func(path_config)
    if missing_annotations:
        missing_annotations_df = pd.DataFrame(missing_annotations)
        missing_annotations_df.to_csv(path_config.missing_annotations_filepath, index=False)
    kld_filepath =path_config.kld_filepath
    # Merge and sort by KLD
    kld_df = pd.read_csv(kld_filepath)
    kld_df.rename(columns={'propbank_predicate': 'Verb', 'kld': 'KLDivergence'}, inplace=True)

    df_summary = df_summary.merge(kld_df, on='Verb', how='left')
    df_summary.sort_values(by='drop in percentage', ascending=False, inplace=True)
    df_summary.to_csv(output_filepath, index=False)



@dataclass
class PathConfig:
    #input
    captions_filepaths: List[Path]
    finalized_captions_filepath: Path
    kld_filepath: Path
    llm_captions_filepath: Path
    vlm_captions_filepath: Path
    #output
    lvl3_output_filepath_for_humans: Path
    lvl3_output_filepath_for_llms: Path
    lvl3_output_filepath_for_vlms: Path
    top_level_filepath_for_humans: Path
    top_level_filepath_for_llms: Path
    top_level_filepath_for_vlms: Path
    mid_level_filepath_for_humans: Path
    mid_level_filepath_for_llms: Path
    mid_level_filepath_for_vlms: Path
    missing_annotations_filepath: Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def main():
    path_config = PathConfig(
        # Input paths
        captions_filepaths=str(DATA_DIR / "results" /"captions_annotated_by_humans"/ "captions_annotated_by_humans.csv"),
        finalized_captions_filepath=str(DATA_DIR / "finalized_captions" / "finalized_captions.csv"),
        kld_filepath=str(DATA_DIR / "kl_divergence" / "propbank_predicate_to_kld_mapping.csv"),
        llm_captions_filepath=str(DATA_DIR / "results" / "llm_annotations" / "captions_annotated_by_llms.csv"),
        vlm_captions_filepath=str(DATA_DIR / "results" / "vlm_annotations" / "images_annotated_by_vlms.csv"),
        # Output paths
        lvl3_output_filepath_for_humans=str(DATA_DIR / "results" /"x_over_20"/ "lvl3"/ "x_over_20_for_humans.csv"),
        lvl3_output_filepath_for_llms=str(DATA_DIR / "results" / "x_over_20"/ "lvl3"/ "x_over_20_for_llms.csv"),
        lvl3_output_filepath_for_vlms=str(DATA_DIR / "results" / "x_over_20"/ "lvl3"/ "x_over_20_for_vlms.csv"),
        top_level_filepath_for_humans=str(DATA_DIR / "results" /"x_over_20"/ "indoor_or_outdoor"/ "x_over_20_for_humans.csv"),
        top_level_filepath_for_llms=str(DATA_DIR / "results" / "x_over_20"/ "indoor_or_outdoor"/ "x_over_20_for_llms.csv"),
        top_level_filepath_for_vlms=str(DATA_DIR / "results" / "x_over_20"/ "indoor_or_outdoor"/ "x_over_20_for_vlms.csv"),
        mid_level_filepath_for_humans=str(DATA_DIR / "results" /"x_over_20"/ "man_made_or_natural"/ "x_over_20_for_humans.csv"),
        mid_level_filepath_for_llms=str(DATA_DIR / "results" / "x_over_20"/ "man_made_or_natural"/ "x_over_20_for_llms.csv"),
        mid_level_filepath_for_vlms=str(DATA_DIR / "results" / "x_over_20"/ "man_made_or_natural"/ "x_over_20_for_vlms.csv"),
        missing_annotations_filepath=str(DATA_DIR / "results" / "missing_annotations_for_human_annotators.csv")
    )
    processor = AnnotationProcessor(path_config)
     # Full classification (lvl3)
    processor.process(load_human_annotations, WORKERS, load_human_output_path, group_by_verb, keep_full_classification, total_per_sentence=20)
    processor.process(load_llm_annotations, LLMS, load_llm_output_path, group_by_verb, keep_full_classification, total_per_sentence=20)
    processor.process(load_vlm_annotations, VLMS, load_vlm_output_path, group_by_verb, keep_full_classification, total_per_sentence=15)

    # Top-level classification (indoor/outdoor)
    processor.process(load_human_annotations, WORKERS, load_top_level_human_output_path, group_by_verb, extract_top_level, total_per_sentence=20)
    processor.process(load_llm_annotations, LLMS, load_top_level_llm_output_path, group_by_verb, extract_top_level, total_per_sentence=20)
    processor.process(load_vlm_annotations, VLMS, load_top_level_vlm_output_path, group_by_verb, extract_top_level, total_per_sentence=15)

    # Mid-level classification (man_made/natural)
    processor.process(load_human_annotations, WORKERS, load_mid_level_human_output_path, group_by_verb, extract_mid_level, total_per_sentence=20)
    processor.process(load_llm_annotations, LLMS, load_mid_level_llm_output_path, group_by_verb, extract_mid_level, total_per_sentence=20)
    processor.process(load_vlm_annotations, VLMS, load_mid_level_vlm_output_path, group_by_verb, extract_mid_level, total_per_sentence=15)

if __name__ == "__main__":
    main()
