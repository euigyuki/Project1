import pandas as pd
import json
from get_fleiss_kappas import bool_dict_to_int_list
import csv
from scipy.spatial.distance import jensenshannon
from helper import categories_to_num_9,normalize_caption,load_combined_df
from helper import clip_probs,categories_to_num_16,nums9_to_categories
from collections import defaultdict
from helper import get_set_of
from helper import WORKERS


def calculate_average_divergence(csv_file):
    df = pd.read_csv(csv_file)
    avg = df['KL Divergence'].mean()
    print(f"Average Jensen-Shannon Divergence in {csv_file}: {avg:.4f}")
    return avg       


class AnnotationProcessor:
    def __init__(self, human_files, llm_files):
        self.human_files = human_files
        self.llm_files = llm_files
        self.human_annotations = {
        "original": {},
        "finalized": {}
         }
        self.llm_annotations = {
        "original": {},
        "finalized": {}
         }

    def process_human_annotations(self, original_captions_set):
        combined_df = load_combined_df(self.human_files)
        print("length of self.human_annotations", len(self.human_annotations),len(combined_df))
        for _, row in combined_df.iterrows():
            workerID = row['WorkerId']
            if row['WorkerId'] == "ASSIGNMENT_ID_NOT_AVAILABLE":
                continue
            if workerID not in WORKERS:
                continue
            caption = normalize_caption(row['Input.sentence'])
            answer_dict = json.loads(row['Answer.taskAnswers'])[0]
            category = self._process_categories(answer_dict['category'])
            location = "indoors" if bool_dict_to_int_list(answer_dict['location']) == 0 else "outdoors"
            type_ = "man-made" if bool_dict_to_int_list(answer_dict['type']) == 0 else "natural"
            total = f"{location}/{type_}/{category}"
            if caption in original_captions_set:
                self.human_annotations["original"].setdefault(caption, []).append(total)
            else:
                self.human_annotations["finalized"].setdefault(caption, []).append(total)
        print("length of self.human_annotations", len(self.human_annotations["original"]),len(self.human_annotations["finalized"])) 

            

    def process_llm_annotations(self, original_captions_set):
        for filepath in self.llm_files:
            with open(filepath, 'r') as file:
                data = json.load(file)
                for item in data:
                    caption = normalize_caption(item['caption'])
                    annotation = item['annotation']
                    category = categories_to_num_9.get(annotation[2], -1)
                    category = nums9_to_categories[category]
                    location = "indoors" if 'indoors' in annotation else "outdoors"
                    type_ = "man-made" if 'man-made' in annotation else "natural"
                    total = f"{location}/{type_}/{category}"
                    if total == "outdoors/natural/recreation":
                        total = "outdoors/man-made/recreation"
                    if caption in original_captions_set:
                        self.llm_annotations["original"].setdefault(caption, []).append(total)
                    else:
                        self.llm_annotations["finalized"].setdefault(caption, []).append(total)

    def _process_categories(self, categories_map):
        for key, value in categories_map.items():
            if value:
                return nums9_to_categories[int(key)]


class DivergenceCalculator:
    @staticmethod
    def calculate_probability_distribution(annotations):
        total = [0] * 16
        for annotation in annotations:
            index = categories_to_num_16[annotation]
            total[index] += 1
        return total

    @staticmethod
    def calculate_jensen_shannon_divergence(human_annotations, llm_annotations):
        divergences = {}
        for caption in human_annotations:
            if caption in llm_annotations:
                human_probs = DivergenceCalculator.calculate_probability_distribution(human_annotations[caption])
                llm_probs = DivergenceCalculator.calculate_probability_distribution(llm_annotations[caption])
                human_probs = clip_probs(human_probs)
                llm_probs = clip_probs(llm_probs)
                js_div = jensenshannon(human_probs, llm_probs)
                divergences[caption] = js_div
        return divergences
    
    @classmethod
    def calculate_all(cls, human_annotations, llm_annotations):
        results = {}
        for split in ['original', 'finalized']:
            results[split] = cls.calculate_jensen_shannon_divergence(
                human_annotations[split],
                llm_annotations[split]
            )
        return results


class FileManager:
    @staticmethod
    def output_to_csv(data, output_csv):
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Caption', 'Jensen-Shannon Divergence'])
            for caption, divergence in data.items():
                writer.writerow([caption, divergence])
        print(f"Data has been written to {output_csv}")

    @staticmethod
    def calculate_average_divergence(csv_file):
        df = pd.read_csv(csv_file)
        avg = df['Jensen-Shannon Divergence'].mean()
        print(f"Average Jensen-Shannon Divergence in {csv_file}: {avg:.4f}")
        return avg
  

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

def get_dict_of(df, value):
    combined_df = load_combined_df(df)
    result = {}
    for _, row in combined_df.iterrows():
        caption = normalize_caption(row[value])
        lemma = row['Lemma']
        result[caption] = lemma
    return result

def group_by_verb(filepaths,js,original_or_finalized):
    # lemmas = get_set_of(filepaths, "Lemma") 
    checked_lemmas = get_dict_of(filepaths, original_or_finalized)
    #result = defaultdict(lambda: defaultdict(float))
    result = defaultdict(lambda: defaultdict(list))
    for caption in checked_lemmas:
        caption = normalize_caption(caption)
        lemma = checked_lemmas[caption]
        result[lemma][caption].append(js[caption])
    return result


def write_grouped_to_csv(nested_dict, output_csv):
    # Prepare to compute sums and counts for averages per outer_key
    sum_dict = defaultdict(float)
    count_dict = defaultdict(int)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['outer_key', 'inner_key', 'value'])  # header

        for outer_key, inner_dict in nested_dict.items():
            for inner_key, value_list in inner_dict.items():
                for val in value_list:  # loop over JS values in the list
                    writer.writerow([outer_key, inner_key, val])
                    sum_dict[outer_key] += val
                    count_dict[outer_key] += 1

        # Add a blank row and then write the averages per outer_key
        writer.writerow([])
        writer.writerow(['outer_key', 'average_value'])  # header for averages
        averages = [(key, sum_dict[key] / count_dict[key]) for key in sum_dict]

        # Sort by average value (ascending). Use `reverse=True` for descending.
        averages.sort(key=lambda x: x[1])  # or key=lambda x: x[1], reverse=True

        # Write sorted averages
        for key, avg in averages:
            writer.writerow([key, f"{avg:.2f}"])

    print(f"Grouped dictionary and outer_key averages have been written to {output_csv}")
 
def main():
    ### Inputs
    captions_filepaths = ["../../data/results/captions/captions.csv"]
    finalized_captions_filepaths = ["../../data/finalized_captions/finalized_captions.csv"]
    llm_annotations_filepaths = [
        "../../data/results/llm_annotations/deepseek_annotations.json",
        "../../data/results/llm_annotations/perplexity_annotations.json",
        "../../data/results/llm_annotations/claude_annotations.json",
        "../../data/results/llm_annotations/chatgpt_annotations.json"
    ]

    ### Outputs
    original_js_output_csv = "../../data/results/original_caption_to_jensenshannon_divergences.csv"
    finalized_js_output_csv = "../../data/results/finalized_caption_to_jensenshannon_divergences.csv"
    original_captions_grouped_by_verb_csv = "../../data/results/original_captions_grouped_by_verb.csv"
    finalized_captions_grouped_by_verb_csv = "../../data/results/finalized_captions_grouped_by_verb.csv"

    # Load original caption set
    original_captions_set = get_set_of(finalized_captions_filepaths, "Original Sentence")

    # Process annotations
    processor = AnnotationProcessor(captions_filepaths, llm_annotations_filepaths)
    processor.process_human_annotations(original_captions_set)
    processor.process_llm_annotations(original_captions_set) #i understand this part

    # Calculate divergences
    results = DivergenceCalculator.calculate_all(
        processor.human_annotations,
        processor.llm_annotations
    )
    original_captions_js = results["original"]
    finalized_captions_js = results["finalized"]

    # Output to CSV and calculate averages
    file_manager = FileManager()
    file_manager.output_to_csv(original_captions_js, original_js_output_csv)
    file_manager.output_to_csv(finalized_captions_js, finalized_js_output_csv)
    file_manager.calculate_average_divergence(original_js_output_csv)
    file_manager.calculate_average_divergence(finalized_js_output_csv)

    # Group by verb
    result = group_by_verb(finalized_captions_filepaths, original_captions_js, "Original Sentence")
    write_grouped_to_csv(result, original_captions_grouped_by_verb_csv)

    result = group_by_verb(finalized_captions_filepaths, finalized_captions_js, "Finalized Sentence")
    write_grouped_to_csv(result, finalized_captions_grouped_by_verb_csv)

#original code
    # original_captions_js,finalized_captions_js = analyze_annotations(finalized_captions_filepaths,captions_filepaths, llm_annotations_filepaths)
    # output_to_csv(original_captions_js, original_js_output_csv)
    # output_to_csv(finalized_captions_js, finalized_js_output_csv)
    # calculate_average_divergence(original_js_output_csv)
    # calculate_average_divergence(finalized_js_output_csv)

    # result =group_by_verb(finalized_captions_filepaths,original_captions_js)
    # write_grouped_to_csv(result, original_captions_grouped_by_verb_csv)

    # result =group_by_verb(finalized_captions_filepaths,original_captions_js)
    # write_grouped_to_csv(result, finalized_captions_grouped_by_verb_csv)

if __name__ == "__main__":
    main()
