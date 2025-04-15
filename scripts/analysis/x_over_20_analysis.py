import pandas as pd
from collections import defaultdict
import json
from helper import WORKERS,nums9_to_categories,normalize_caption

EXPECTED_ANNOTATIONS = 10
TOTAL_PER_SENTENCE = 20

class ClassificationMapper:
    def __init__(self, picked_captions_filepath):
        self.verb_to_classification = defaultdict(str)
        self.sentences_to_classification = defaultdict(str)
        self.processed_sentence_to_classification = defaultdict(str)
        self.sentences_to_verbs = defaultdict(str)
        self.verbs_to_original = defaultdict(list)
        self.verbs_to_processed = defaultdict(list)
        self._load_data(picked_captions_filepath)

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






def counting_for_analysis(filepaths,picked_captions_filepath,missing_annotations_filepath,output_filepath):
    """
    Processes CSV files and counts verb occurrences per classification.
    """
    mapper  = ClassificationMapper(picked_captions_filepath)
    combination = pd.concat([pd.read_csv(fp) for fp in filepaths])
    selected_columns = combination[['WorkerId', 'Input.sentence', 'Answer.taskAnswers']]
    selected_columns = selected_columns[selected_columns['WorkerId'].isin(WORKERS)]
    
    selected_columns['verbs'] = selected_columns['Input.sentence'].apply(
        lambda x: mapper.get_verb(x)
    )
    
    df = pd.DataFrame(selected_columns)
    df.to_csv('sanity_check.csv', index=False)
    missing_annotations = []
    summary_rows = []
    grouped = selected_columns.groupby('verbs')
    for verb, group in grouped:
        print(f"Group: {verb}, Size: {len(group)}")
        # Count annotations per worker for this verb
        per_worker_count = group['WorkerId'].value_counts()
        for worker in WORKERS:
            count = per_worker_count.get(worker, 0)
            # print(f"   {worker}: {count} / {EXPECTED_ANNOTATIONS}")

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
            # Save missing annotation data to CSV for debugging
        if missing_annotations:
            missing_df = pd.DataFrame(missing_annotations)
            missing_df.to_csv(missing_annotations_filepath, index=False)

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(output_filepath, index=False)


def main():
    captions_filepaths = ["../../data/results/captions/captions.csv"]

    # images_filepaths = ["/app/data/results/images1.csv", "/app/data/results/images2.csv"]
    finalized_captions_filepath = "../../data/finalized_captions/finalized_captions.csv"


    #output
    output_filepath = "../../data/results/x_over_20.csv"
    missing_annotations_filepath = "../../data/results/missing_annotations.csv"

    counting_for_analysis(captions_filepaths,finalized_captions_filepath,missing_annotations_filepath,output_filepath)

if __name__ == "__main__":
    main()
