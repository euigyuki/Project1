import pandas as pd
from collections import defaultdict
import json
from dataclasses import dataclass
from typing import List
from scripts.helper.helper_functions import normalize_caption,load_combined_df, WORKERS  
from scripts.helper.helper_functions import nums9_to_categories
from pathlib import Path

EXPECTED_ANNOTATIONS = 10
TOTAL_PER_SENTENCE = 20
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

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
            classification = ":".join(
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

    def get_classification(self, sentence):
        sentence = normalize_caption(sentence)
        return (self.sentences_to_classification.get(sentence) or 
                self.processed_sentence_to_classification.get(sentence))

    def get_verb(self, sentence):
        return self.sentences_to_verbs.get(normalize_caption(sentence), '')
    
    @staticmethod
    def create_classification(answer_list):
        merged = {key: value for d in answer_list for key, value in d.items()}
        category = merged.get('category', {})
        indoors_outdoors = merged.get('location', {})
        man_made_or_natural = merged.get('type', {})
        valid = [k for k, v in indoors_outdoors.items() if v]
        valid += [k for k, v in man_made_or_natural.items() if v]
        valid += [nums9_to_categories[int(k)] for k, v in category.items() if v]
        return ":".join(valid)

def filter_valid_annotations(df, workers):
    df = df[df['WorkerId'].isin(workers)]
    return df[['WorkerId', 'Input.sentence', 'Answer.taskAnswers']]


def group_annotations_by_verb(df, mapper):
    df = df.copy()
    df['verbs'] = df['Input.sentence'].apply(lambda x: mapper.get_verb(x))
    grouped = df.groupby('verbs')
    return grouped, df

def generate_annotation_report(path_config):
    """
    Processes CSV files and counts verb occurrences per classification.
    """
    caption_filepaths = path_config.captions_filepaths
    finalized_captions_filepath = path_config.finalized_captions_filepath
    
    mapper  = ClassificationMapper(finalized_captions_filepath)
    combination = load_combined_df(caption_filepaths)
    selected_columns = filter_valid_annotations(combination, WORKERS)
    grouped, selected_columns = group_annotations_by_verb(selected_columns, mapper)
    
    
    missing_annotations = []
    summary_rows = []
    print(grouped)
    for verb, group in grouped:
        print(f"Group: {verb}, Size: {len(group)}")
        # Count annotations per worker for this verb
        per_worker_count = group['WorkerId'].value_counts()
        for worker in WORKERS:
            count = per_worker_count.get(worker, 0)

            if count < EXPECTED_ANNOTATIONS:
                missing_annotations.append({
                    "WorkerId": worker,
                    "Verb": verb,
                    "Provided": count,
                    "Missing": EXPECTED_ANNOTATIONS - count
                })

        print("Annotations per worker:")

        original_count, processed_count = 0, 0
        for _, row in group.iterrows():
            sentence = normalize_caption(row['Input.sentence'])
            classification_of_annotator = row['Answer.taskAnswers']
            answer_dict = json.loads(row['Answer.taskAnswers'])
            classification_of_annotator = mapper.create_classification(answer_dict)
            ground_truth = mapper.get_classification(sentence)

            if ground_truth == classification_of_annotator:
                if sentence in mapper.sentences_to_classification:
                    original_count += 1
                elif sentence in mapper.processed_sentence_to_classification:
                    processed_count += 1
        original_count_percentage = original_count / TOTAL_PER_SENTENCE
        processed_count_percentage = processed_count / TOTAL_PER_SENTENCE
        print(f"Original count: {original_count}, Processed count: {processed_count}")
        print(f"Original count percentage: {original_count_percentage}")
        print(f"Processed count percentage: {processed_count_percentage}")
        print()
        summary_rows.append({
            "Verb": verb,
            "Total annotations": len(group),
            "Worker A17EZEAMF37MGQ": f"{per_worker_count.get('A17EZEAMF37MGQ', 0)} / {EXPECTED_ANNOTATIONS}",
            "Worker A176JUTGNWG7QJ": f"{per_worker_count.get('A176JUTGNWG7QJ', 0)} / {EXPECTED_ANNOTATIONS}",
            "Worker A2SMHEGRLML092": f"{per_worker_count.get('A2SMHEGRLML092', 0)} / {EXPECTED_ANNOTATIONS}",
            "Worker A2ZY94PZ5CVH0": f"{per_worker_count.get('A2ZY94PZ5CVH0', 0)} / {EXPECTED_ANNOTATIONS}",
            "Original correct": original_count,
            "Processed correct": processed_count,
            "Original percentage": original_count_percentage,
            "Processed percentage": processed_count_percentage,
        })
    
    df_summary = pd.DataFrame(summary_rows)
    return df_summary,missing_annotations
    
def save_summary(path_config, df_summary,missing_annotations_human):
    if missing_annotations_human:
        missing_annotations_df = pd.DataFrame(missing_annotations_human)
        missing_annotations_df.to_csv(path_config.missing_annotations_filepath, index=False)
    kld_filepath =path_config.kld_filepath
    output_filepath = path_config.output_filepath_for_human_annotators
    # Merge and sort by KLD
    kld_df = pd.read_csv(kld_filepath)
    kld_df.rename(columns={'propbank_predicate': 'Verb', 'kld': 'KLDivergence'}, inplace=True)

    df_summary = df_summary.merge(kld_df, on='Verb', how='left')
    df_summary.sort_values(by='KLDivergence', ascending=False, inplace=True)
    df_summary.to_csv(output_filepath, index=False)


@dataclass
class PathConfig:
    captions_filepaths: List[str]
    finalized_captions_filepath: str
    kld_filepath: str
    output_filepath_for_human_annotators: str
    output_filepath_for_llms: str
    missing_annotations_filepath: str

def main():
    path_config = PathConfig(
        # Input paths
        captions_filepaths=[str(DATA_DIR / "results" /"captions_annotated_by_humans"/ "captions_annotated_by_humans_backup.csv")],
        finalized_captions_filepath=str(DATA_DIR / "finalized_captions" / "finalized_captions.csv"),
        kld_filepath=str(DATA_DIR / "kl_divergence" / "propbank_predicate_to_kld_mapping.csv"),
        # Output paths
        output_filepath_for_human_annotators=str(DATA_DIR / "results" / "x_over_20_for_human_annotators.csv"),
        output_filepath_for_llms=str(DATA_DIR / "results" / "x_over_20_for_llms.csv"),
        missing_annotations_filepath=str(DATA_DIR / "results" / "missing_annotations_for_human_annotators.csv")
    )
    df_human,missing_annotations_human = generate_annotation_report(path_config)
    save_summary(path_config, df_human,missing_annotations_human)
    
if __name__ == "__main__":
    main()
