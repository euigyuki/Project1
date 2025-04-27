import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from typing import List
from scripts.helper.helper_functions import normalize_caption, WORKERS  
from scripts.helper.helper_functions import nums9_to_categories,LLMS,VLMS
from pathlib import Path

EXPECTED_ANNOTATIONS = 10
TOTAL_PER_SENTENCE = 20


class ClassificationMapper:
    def __init__(self, finalized_captions_filepath):
        self.verb_to_classification = defaultdict(str)
        self.sentences_to_classification = defaultdict(str)
        self.processed_sentence_to_classification = defaultdict(str)
        self.sentences_to_verbs = defaultdict(str)
        self.verbs_to_original = defaultdict(list)
        self.verbs_to_processed = defaultdict(list)
        self._load_data(finalized_captions_filepath)

    def _load_data(self, filepath):
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            classification = "/".join(
                str(val) for val in [row['q1'], row['q2'], row['q3'], row['q4']] if pd.notna(val)
            )
            verb = row['Verb']
            sentence = normalize_caption(row['Sentence'])
            finalized = normalize_caption(row['Finalized Sentence'])

            self.verb_to_classification[verb] = classification
            self.sentences_to_verbs[sentence] = verb
            self.sentences_to_verbs[finalized] = verb

            self.verbs_to_original[verb].append(sentence)
            self.verbs_to_processed[verb].append(finalized)

            self.sentences_to_classification[sentence] = classification
            self.processed_sentence_to_classification[finalized] = classification
        print(len(self.sentences_to_classification))
        print(len(self.processed_sentence_to_classification))

    def get_classification(self, sentence):
        sentence = normalize_caption(sentence)
        return (self.sentences_to_classification.get(sentence) or 
                self.processed_sentence_to_classification.get(sentence))

    def get_verb(self, sentence):
        result = self.sentences_to_verbs.get(normalize_caption(sentence), '')
        return result
    
    @staticmethod
    def create_classification(answer_list):
        merged = {key: value for d in answer_list for key, value in d.items()}
        category = merged.get('category', {})
        indoors_outdoors = merged.get('location', {})
        man_made_or_natural = merged.get('type', {})
        valid = [k for k, v in indoors_outdoors.items() if v]
        valid += [k for k, v in man_made_or_natural.items() if v]
        valid += [nums9_to_categories[int(k)] for k, v in category.items() if v]
        return "/".join(valid)

    @staticmethod
    def create_location_only_classification(answer_list):
        """
        Extracts only the location (indoor/outdoor) labels from the answer list.
        """
        merged = {key: value for d in answer_list for key, value in d.items()}
        indoors_outdoors = merged.get('location', {})
        return indoors_outdoors

class AnnotationProcessor:
    def __init__(self, path_config):
        self.path_config = path_config

    def process(self, loader_func, agents, output_func):
        df_summary, missing = generate_annotation_report(self.path_config, loader_func, agents)
        save_summary(self.path_config, df_summary, missing, output_func)

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

def compute_correct_counts(group, mapper):
    """
    Count how many annotations match the classification (original vs. processed).
    """
    original_count, processed_count = 0, 0
    for _, row in group.iterrows():
        sentence = normalize_caption(row['Input.sentence'])
        annotation = row['Answer.taskAnswers']
        ground_truth = mapper.get_classification(sentence)
        if ground_truth == annotation:
            if sentence in mapper.sentences_to_classification:
                original_count += 1
            elif sentence in mapper.processed_sentence_to_classification:
                processed_count += 1
    return original_count, processed_count

def generate_annotation_report(path_config, load_annotations_func, WORKERS_or_LLMS):
    caption_filepath = load_annotations_func(path_config)
    finalized_captions_filepath = path_config.finalized_captions_filepath

    mapper = ClassificationMapper(finalized_captions_filepath)
    combination = pd.read_csv(caption_filepath)
    selected_columns = filter_valid_annotations(combination, WORKERS_or_LLMS)
    selected_columns = selected_columns.drop_duplicates(subset=["WorkerId", "Input.sentence"])
    grouped, selected_columns = group_annotations_by_verb(selected_columns, mapper)

    summary_rows = []
    all_missing = []

    for verb, group in grouped:
        # print(f"Group: {verb}, Size: {len(group)}")
        row, missing = summarize_group(verb, group, WORKERS_or_LLMS, mapper)
        summary_rows.append(row)
        all_missing.extend(missing)

    df_summary = pd.DataFrame(summary_rows)
    return df_summary, all_missing


def filter_valid_annotations(df, workers):
    df = df[df['WorkerId'].isin(workers)]
    return df[['WorkerId', 'Input.sentence', 'Answer.taskAnswers']]


def group_annotations_by_verb(df, mapper):
    df = df.copy()
    df['verbs'] = df['Input.sentence'].apply(lambda x: mapper.get_verb(x))
    print(df['verbs'].unique(),"unique verbs")
    grouped = df.groupby('verbs')
    return grouped, df


def summarize_group(verb, group, workers, mapper):
    missing, counts = compute_worker_missing(group, workers, verb)
    original_correct, processed_correct = compute_correct_counts(group, mapper)
    original_pct = original_correct / TOTAL_PER_SENTENCE
    processed_pct = processed_correct / TOTAL_PER_SENTENCE
    row = {
        "Verb": verb,
        "Total annotations": len(group),
        "Original correct": original_correct,
        "Processed correct": processed_correct,
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
    return path_config.lvl3_output_filepath_for_human_annotators

def load_llm_output_path(path_config):
    return path_config.lvl3_output_filepath_for_llms

def load_vlm_output_path(path_config):
    return path_config.lvl3_output_filepath_for_vlms


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
    df_summary.sort_values(by='KLDivergence', ascending=False, inplace=True)
    df_summary.to_csv(output_filepath, index=False)



@dataclass
class PathConfig:
    #input
    captions_filepaths: List[str]
    finalized_captions_filepath: Path
    kld_filepath: Path
    llm_captions_filepath: Path
    vlm_captions_filepath: Path
    #output
    lvl3_output_filepath_for_human_annotators: Path
    lvl3_output_filepath_for_llms: Path
    lvl3_output_filepath_for_vlms: Path
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
        lvl3_output_filepath_for_human_annotators=str(DATA_DIR / "results" /"x_over_20"/ "lvl3"/ "x_over_20_for_human_annotators.csv"),
        lvl3_output_filepath_for_llms=str(DATA_DIR / "results" / "x_over_20"/ "lvl3"/ "x_over_20_for_llms.csv"),
        lvl3_output_filepath_for_vlms=str(DATA_DIR / "results" / "x_over_20"/ "lvl3"/ "x_over_20_for_vlms.csv"),
        missing_annotations_filepath=str(DATA_DIR / "results" / "missing_annotations_for_human_annotators.csv")
    )
    processor = AnnotationProcessor(path_config)

    for loader, group, output_path_func in [
        (load_human_annotations, WORKERS, load_human_output_path),
        (load_llm_annotations, LLMS, load_llm_output_path),
        (load_vlm_annotations, VLMS, load_vlm_output_path),
    ]:
        processor.process(loader, group, output_path_func)

if __name__ == "__main__":
    main()
