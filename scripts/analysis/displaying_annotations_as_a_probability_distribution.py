import pandas as pd
import json
from get_fleiss_kappas import bool_dict_to_int_list
import csv
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from Project1.scripts.helper.helper_functions import categories_to_num_9,normalize_caption,load_combined_df
from Project1.scripts.helper.helper_functions import clip_probs,categories_to_num_16,nums9_to_categories
from collections import defaultdict
from Project1.scripts.helper.helper_functions import get_set_of
from Project1.scripts.helper.helper_functions import WORKERS
import numpy as np
from Project1.scripts.helper.helper_functions import AnnotationProcessor



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
                prev_human_probs = DivergenceCalculator.calculate_probability_distribution(human_annotations[caption])
                prev_llm_probs = DivergenceCalculator.calculate_probability_distribution(llm_annotations[caption])
                human_probs = clip_probs(prev_human_probs)
                llm_probs = clip_probs(prev_llm_probs)
                js_div = jensenshannon(human_probs, llm_probs)
                kl_div_human_llm = np.sum(rel_entr(np.array(human_probs),np.array(llm_probs)))
                kl_div_llm_human = np.sum(rel_entr(np.array(llm_probs),np.array(human_probs)))
                divergences[caption] = {
                "kl_div_human_llm": kl_div_human_llm,
                "kl_div_llm_human": kl_div_llm_human,
                "js_div": js_div,
                "human_probs": prev_human_probs,
                "llm_probs": prev_llm_probs
            }
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
            writer.writerow(['Caption', 'KLD Human to LLM', 'KLD LLM to Human',
                             'Jensen-Shannon Divergence', 'Human Probs', 'LLM Probs'])
            for caption, divergence in data.items():
                writer.writerow([
                    caption,
                    divergence["kl_div_human_llm"],
                    divergence["kl_div_llm_human"],
                    divergence['js_div'],  # only write the numeric value here
                    json.dumps(divergence['human_probs']),
                    json.dumps(divergence['llm_probs'])
                ])
        print(f"Data has been written to {output_csv}")

    @staticmethod
    def calculate_average_divergence(csv_file):
        df = pd.read_csv(csv_file)
        avg = df['Jensen-Shannon Divergence'].mean()
        print(f"Average Jensen-Shannon Divergence in {csv_file}: {avg:.4f}")
        return avg

def get_dict_of(df, value):
    combined_df = load_combined_df(df)
    result = {}
    for _, row in combined_df.iterrows():
        caption = normalize_caption(row[value])
        lemma = row['Verb']
        result[caption] = lemma
    return result

def group_by_verb(filepaths, js_dict, original_or_finalized):
    checked_lemmas = get_dict_of(filepaths, original_or_finalized)
    result = defaultdict(lambda: defaultdict(list))
    for caption in checked_lemmas:
        norm_caption = normalize_caption(caption)
        lemma = checked_lemmas[caption]
        if norm_caption in js_dict:
            result[lemma][norm_caption].append(js_dict[norm_caption])  # dict with js_div, human_probs, llm_probs
    return result




def write_grouped_to_csv(nested_dict, output_csv):
    sum_jsd = defaultdict(float)
    sum_kld_human_llm = defaultdict(float)
    sum_kld_llm_human = defaultdict(float)
    count_dict = defaultdict(int)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'outer_key', 'inner_key', 'KLD Human to LLM',
            'KLD LLM to Human',
            'Jensen-Shannon Divergence',
            'Human Probabilities', 'LLM Probabilities'
        ])  # header for detailed rows

        for outer_key, inner_dict in nested_dict.items():
            for inner_key, annotation_list in inner_dict.items():
                for annotation in annotation_list:
                    kl_div_human_llm = annotation['kl_div_human_llm']
                    kl_div_llm_human = annotation['kl_div_llm_human']
                    js_div = annotation['js_div']
                    human_probs = annotation['human_probs']
                    llm_probs = annotation['llm_probs']

                    writer.writerow([
                        outer_key, inner_key,
                        kl_div_human_llm, kl_div_llm_human, js_div,
                        json.dumps(human_probs), json.dumps(llm_probs)
                    ])

                    sum_kld_human_llm[outer_key] += kl_div_human_llm
                    sum_kld_llm_human[outer_key] += kl_div_llm_human
                    sum_jsd[outer_key] += js_div
                    count_dict[outer_key] += 1

        # Blank row before summary
        writer.writerow([])
        writer.writerow(['outer_key', 'average_KLD_Human_to_LLM', 'average_KLD_LLM_to_Human', 'average_JSD'])

        all_keys = sorted(count_dict.keys(), key=lambda k: sum_jsd[k] / count_dict[k])  # sort by JSD

        for key in all_keys:
            avg_kld_human_llm = sum_kld_human_llm[key] / count_dict[key]
            avg_kld_llm_human = sum_kld_llm_human[key] / count_dict[key]
            avg_jsd = sum_jsd[key] / count_dict[key]
            writer.writerow([
                key,
                f"{avg_kld_human_llm:.2f}",
                f"{avg_kld_llm_human:.2f}",
                f"{avg_jsd:.2f}"
            ])

    print(f"Grouped dictionary with full probability info written to {output_csv}")


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

if __name__ == "__main__":
    main()
