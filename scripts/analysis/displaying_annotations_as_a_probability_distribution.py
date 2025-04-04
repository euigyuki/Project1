import pandas as pd
import json
from get_fleiss_kappas import bool_dict_to_int_list
import csv
from scipy.spatial.distance import jensenshannon
from helper import categories_to_num_9,normalize_caption,load_combined_df
from helper import clip_probs,categories_to_num_16,nums9_to_categories
from collections import defaultdict
from helper import get_set_of


def process_categories(categories_map):
    for key, value in categories_map.items():
        if value:
            return nums9_to_categories[int(key)]


# Function to calculate probability distribution
def calculate_probability_distribution(annotations):
    total = [0]*16
    for annotation in annotations:
        index = categories_to_num_16[annotation]
        total[index] += 1
    return total

def calculate_average_divergence(csv_file):
    df = pd.read_csv(csv_file)
    avg = df['KL Divergence'].mean()
    print(f"Average Jensen-Shannon Divergence in {csv_file}: {avg:.4f}")
    return avg       

# Function to process human annotations
def process_human_caption_annotations(original_captions_set,filepaths):
    combined_df = load_combined_df(filepaths)
    original_annotations = {}
    finalized_annotations = {}
    processed = 0
    skipped = 0
    for _, row in combined_df.iterrows():
        if row['WorkerId'] == "ASSIGNMENT_ID_NOT_AVAILABLE":
            continue
        caption = row['Input.sentence']
        caption = normalize_caption(caption)
        answer_dict = json.loads(row['Answer.taskAnswers'])[0]
        category = process_categories(answer_dict['category'])
        location = "indoors" if bool_dict_to_int_list(answer_dict['location'])==0 else "outdoors"
        type_ = "man-made" if bool_dict_to_int_list(answer_dict['type']) == 0 else "natural"
        total = location+"/" + type_+"/"+category
        if caption in original_captions_set:
            original_annotations.setdefault(caption, []).append(total)
            processed += 1
        else:
            finalized_annotations.setdefault(caption, []).append(total)
            skipped += 1
    print(f"[Human Annotations] Processed: {processed}, Skipped: {skipped}")
    return original_annotations, finalized_annotations



# Function to process LLM annotations
def process_llm_annotations(original_captions_set,filepaths):
    original_annotations = {}
    finalized_annotations = {}
    for filepath in filepaths:
        with open(filepath, 'r') as file:
            data = json.load(file)
            for item in data:
                caption = item['caption']
                caption = normalize_caption(caption)
                annotation = item['annotation']
                category = categories_to_num_9.get(annotation[2], -1)
                category = nums9_to_categories[category]
                location = "indoors" if 'indoors' in annotation else "outdoors"
                type_ = "man-made" if 'man-made' in annotation else "natural"
                total = location+"/" + type_+"/"+category
                if total == "outdoors/natural/recreation":
                    total = "outdoors/man-made/recreation"
                if caption in original_captions_set:
                    original_annotations.setdefault(caption, []).append(total)
                else:
                    finalized_annotations.setdefault(caption, []).append(total)
    return original_annotations, finalized_annotations

    

# Main analysis function
def analyze_annotations(finalized_captions,captions_filepaths, llm_files):
    original_captions_set= get_set_of(finalized_captions,"Original Sentence")
    human_caption_annotations_original,human_caption_annotations_finalized = process_human_caption_annotations(original_captions_set,captions_filepaths)
    llm_annotations_original, llm_annotations_finalized = process_llm_annotations(original_captions_set,llm_files)

    original_captions_jensen_shannon_divergences = {}
    finalized_captions_jensen_shannon_divergences = {}

    for caption in human_caption_annotations_original:
        if caption in llm_annotations_original:
            human_probs = calculate_probability_distribution(human_caption_annotations_original[caption])
            llm_probs = calculate_probability_distribution(llm_annotations_original[caption])
            # Ensure no zero probabilities to avoid issues in KL divergence calculation
            human_probs = clip_probs(human_probs)
            llm_probs = clip_probs(llm_probs)
            # Calculate Jensen-Shannon divergence
            js_div = jensenshannon(human_probs, llm_probs)
            original_captions_jensen_shannon_divergences[caption] = js_div
    for caption in human_caption_annotations_finalized:
        if caption in llm_annotations_finalized:
            human_probs = calculate_probability_distribution(human_caption_annotations_finalized[caption])
            llm_probs = calculate_probability_distribution(llm_annotations_finalized[caption])
            # Ensure no zero probabilities to avoid issues in KL divergence calculation
            human_probs = clip_probs(human_probs)
            llm_probs = clip_probs(llm_probs)
            js_div = jensenshannon(human_probs, llm_probs)
            finalized_captions_jensen_shannon_divergences[caption] = js_div
    return original_captions_jensen_shannon_divergences,finalized_captions_jensen_shannon_divergences

  

def output_to_csv(divergences, output_csv):
      # Write the KL divergences to the CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Caption', 'KL Divergence'])
        # Write each caption and its corresponding KL divergence
        for caption, div in divergences.items():
            writer.writerow([caption, div])
    print(f"jensen shannon divergences have been written to {output_csv}")        


def group_by_verb(df,js):
    verbs = get_set_of(df, "Lemma") 
    result = defaultdict(lambda: defaultdict(float))
    for verb in verbs:
        for caption in js:
            if verb in caption:
                result[verb][caption] = js[caption]
    return result


def write_grouped_to_csv(nested_dict, output_csv):
    # Prepare to compute sums and counts for averages per outer_key
    sum_dict = defaultdict(float)
    count_dict = defaultdict(int)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['outer_key', 'inner_key', 'value'])  # header

        for outer_key, inner_dict in nested_dict.items():
            for inner_key, value in inner_dict.items():
                writer.writerow([outer_key, inner_key, value])
                sum_dict[outer_key] += value
                count_dict[outer_key] += 1

        # Add a blank row and then write the averages per outer_key
        writer.writerow([])
        writer.writerow(['outer_key', 'average_value'])  # header for averages

        for key in sorted(sum_dict):  # optional: sort outer keys
            avg = sum_dict[key] / count_dict[key]
            writer.writerow([key, f"{avg:.2f}"])

    print(f"Grouped dictionary and outer_key averages have been written to {output_csv}")
 
def main():

    ###inputs
    captions_filepaths = ["../../data/results/captions/captions1.csv", "../../data/results/captions/captions2.csv"]
    finalized_captions_filepaths = ["../../data/finalized_captions/finalized_captions.csv"]
    llm_annotations_filepaths = [
    "../../data/results/llm_annotations/deepseek_annotations.json",
    "../../data/results/llm_annotations/perplexity_annotations.json",
    "../../data/results/llm_annotations/claude_annotations.json",
    "../../data/results/llm_annotations/chatgpt_annotations.json"
    ]
    ###output
    original_js_output_csv = "../../data/results/original_caption_to_jensenshannon_divergences.csv"
    finalized_js_output_csv = "../../data/results/finalized_caption_to_jensenshannon_divergences.csv"
    original_captions_grouped_by_verb_csv = "../../data/results/original_captions_grouped_by_verb.csv"
    finalized_captions_grouped_by_verb_csv = "../../data/results/finalized_captions_grouped_by_verb.csv"

    original_captions_js,finalized_captions_js = analyze_annotations(finalized_captions_filepaths,captions_filepaths, llm_annotations_filepaths)
    output_to_csv(original_captions_js, original_js_output_csv)
    output_to_csv(finalized_captions_js, finalized_js_output_csv)
    calculate_average_divergence(original_js_output_csv)
    calculate_average_divergence(finalized_js_output_csv)

    result =group_by_verb(finalized_captions_filepaths,original_captions_js)
    write_grouped_to_csv(result, original_captions_grouped_by_verb_csv)

    result =group_by_verb(finalized_captions_filepaths,original_captions_js)
    write_grouped_to_csv(result, finalized_captions_grouped_by_verb_csv)

if __name__ == "__main__":
    main()
