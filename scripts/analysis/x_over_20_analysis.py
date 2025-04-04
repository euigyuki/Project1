import pandas as pd
from collections import defaultdict
import json
from collections import Counter
from helper import WORKERS,categories

EXPECTED_ANNOTATIONS = 10


def normalize_and_strip_quotes(sentence):
    """
    Normalize and clean up sentences by replacing typographic quotes and stripping extra whitespace.
    """
    return sentence.replace('“', '"').replace('”', '"').strip('"').strip()


def create_classification(answer_list):
    """
    Creates a classification string based on the provided answer list.
    """
    # Merge all dictionaries in the list into a single dictionary
    merged_dict = {key: value for d in answer_list for key, value in d.items()}

    # Extract relevant sub-dictionaries
    category = merged_dict.get('category', {})
    indoors_outdoors = merged_dict.get('location', {})
    man_made_or_natural = merged_dict.get('type', {})

    # Collect valid location keys
    valid_values = [k for k, v in indoors_outdoors.items() if v]
    valid_values.extend([k for k, v in man_made_or_natural.items() if v])
    valid_values.extend([categories[int(k)] for k, v in category.items() if v])

    # Create classification string
    classification = ":".join(valid_values)
    return classification



def create_classifications(picked_captions_filepath):
    """
    Reads data from CSV and generates mappings for verb classification and sentence associations.
    """
    df = pd.read_csv(picked_captions_filepath)
    
    verb_to_classification = defaultdict(str)
    sentences_to_verbs = defaultdict(str)
    sentences_to_classification = defaultdict(str)
    processed_sentence_to_classification = defaultdict(str)
    verbs_to_original_captions = defaultdict(list)
    verbs_to_processed_captions = defaultdict(list)
    
    for _, row in df.iterrows():
        valid_values = [str(val) for val in [row['q1'], row['q2'], row['q3'], row['q4']] if pd.notna(val)]
        classification = ":".join(valid_values)
        
        verb = row['Verb']
        sentence = normalize_and_strip_quotes(row['Sentence'])
        finalized_sentence = normalize_and_strip_quotes(row['Finalized sentence'])
        
        verb_to_classification[verb] = classification
        sentences_to_verbs[sentence] = verb
        sentences_to_verbs[finalized_sentence] = verb
        
        verbs_to_original_captions[verb].append(sentence)
        verbs_to_processed_captions[verb].append(finalized_sentence)
        
        sentences_to_classification[sentence] = classification
        processed_sentence_to_classification[finalized_sentence] = classification
    return verb_to_classification, sentences_to_classification,processed_sentence_to_classification, verbs_to_original_captions, verbs_to_processed_captions, sentences_to_verbs



def counting_for_analysis(filepaths,picked_captions_filepath):
    """
    Processes CSV files and counts verb occurrences per classification.
    """
    verb_to_classification, sentences_to_classification, processed_sentence_to_classification, verbs_to_original_captions, verbs_to_processed_captions, sentences_to_verbs = create_classifications(picked_captions_filepath)
    
    combination = pd.concat([pd.read_csv(fp) for fp in filepaths])
    selected_columns = combination[['WorkerId', 'Input.sentence', 'Answer.taskAnswers']]
    selected_columns = selected_columns[selected_columns['WorkerId'].isin(WORKERS)]
    


    selected_columns['verbs'] = selected_columns['Input.sentence'].apply(
        lambda x: sentences_to_verbs.get(normalize_and_strip_quotes(x), '')
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
            print(f"   {worker}: {count} / {EXPECTED_ANNOTATIONS}")

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
            sentence = normalize_and_strip_quotes(row['Input.sentence'])
            annotator_id = row['WorkerId']
            if sentence in sentences_to_classification:
                answer_dict = json.loads(row['Answer.taskAnswers'])
                classification_of_annotator= create_classification(answer_dict)
                ground_truth = sentences_to_classification.get(sentence, '')
                if ground_truth == classification_of_annotator:
                    original_count += 1
            elif sentence in processed_sentence_to_classification:
                answer_dict = json.loads(row['Answer.taskAnswers'])
                classification_of_annotator= create_classification(answer_dict)
                ground_truth = processed_sentence_to_classification.get(sentence, '')
                if ground_truth == classification_of_annotator:
                    processed_count += 1
        print(f"Original count: {original_count}, Processed count: {processed_count}")
        summary_rows.append({
            "Verb": verb,
            "Total annotations": len(group),
            "Worker A17EZEAMF37MGQ": f"{per_worker_count.get('A17EZEAMF37MGQ', 0)} / {EXPECTED_ANNOTATIONS}",
            "Worker A176JUTGNWG7QJ": f"{per_worker_count.get('A176JUTGNWG7QJ', 0)} / {EXPECTED_ANNOTATIONS}",
            "Worker A2SMHEGRLML092": f"{per_worker_count.get('A2SMHEGRLML092', 0)} / {EXPECTED_ANNOTATIONS}",
            "Worker A2ZY94PZ5CVH0": f"{per_worker_count.get('A2ZY94PZ5CVH0', 0)} / {EXPECTED_ANNOTATIONS}",
            "Original correct": original_count,
            "Processed correct": processed_count
        })
            # Save missing annotation data to CSV for debugging
        if missing_annotations:
            missing_df = pd.DataFrame(missing_annotations)
            missing_df.to_csv("missing_annotations.csv", index=False)
            print("\n⚠️ Missing annotation details saved to 'missing_annotations.csv'")

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv("x_over_20.csv", index=False)

        # print("\n")

def main():
    #captions_filepaths = ["/app/data/results/captions1.csv", "/app/data/results/captions2.csv"]
    captions_filepaths = ["captions1.csv", "captions2.csv"]

    # images_filepaths = ["/app/data/results/images1.csv", "/app/data/results/images2.csv"]
    #picked_captions_filepath = "/app/data/picked_captions/picked_captions.csv"
    picked_captions_filepath = "../picked_captions/picked_captions.csv"

    #output
    output_filepath = "x_over_20.csv"
    counting_for_analysis(captions_filepaths,picked_captions_filepath)

if __name__ == "__main__":
    main()
