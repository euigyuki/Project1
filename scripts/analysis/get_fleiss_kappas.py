import pandas as pd
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
import numpy as np
import json
from Project1.scripts.helper.helper_functions import load_combined_df,WORKERS,categories_to_num_9
from Project1.scripts.helper.helper_functions import bool_dict_to_int_list



         
def calculate_fleiss_kappa(data):
    """
    Calculates Fleiss' Kappa for the provided data.
    """
    aggregated_data, _ = aggregate_raters(data)
    kappa = fleiss_kappa(aggregated_data)
    return kappa



def process_group(group):
    """
    Processes a group of data to extract category, location, and type annotations.
    """
    category_group = []
    indoors_or_outdoors_group = []
    man_made_or_natural_group = []


    for _, row in group.iterrows():
        if row['WorkerId'] not in WORKERS:
            continue

        answer_dict = json.loads(row['Answer.taskAnswers'])
        category = answer_dict[0]['category']
        indoors_or_outdoors = answer_dict[0]['location']
        man_made_or_natural = answer_dict[0]['type']

        category_group.append(bool_dict_to_int_list(category))
        indoors_or_outdoors_group.append(bool_dict_to_int_list(indoors_or_outdoors))
        man_made_or_natural_group.append(bool_dict_to_int_list(man_made_or_natural))

    return category_group, indoors_or_outdoors_group, man_made_or_natural_group

def calculate_fleiss_kappas(categories, indoors_or_outdoors, man_made_or_natural):
    print(f"Fleiss' Kappa - categories: {calculate_fleiss_kappa(categories)}")
    print(f"Fleiss' Kappa - indoors or outdoors: {calculate_fleiss_kappa(indoors_or_outdoors)}")
    print(f"Fleiss' Kappa - man-made or natural: {calculate_fleiss_kappa(man_made_or_natural)}")

def analyze_data(filepaths, group_by_column):
    combined_data = load_combined_df(filepaths)
    selected_columns = combined_data[['WorkerId', group_by_column, 'Answer.taskAnswers']]
    grouped = selected_columns.groupby(group_by_column)

    categories, indoors_or_outdoors, man_made_or_natural = [], [], []
    missing_labels = 0
    for name, group in grouped:
        category_group, location_group, type_group = process_group(group)
        if all(len(lst) == 4 for lst in [category_group, location_group, type_group]):
            categories.append(category_group)
            indoors_or_outdoors.append(location_group)
            man_made_or_natural.append(type_group)
        else:
            print(f"Skipping group {name} due to inconsistent lengths.")
            print(f"Lengths: {len(category_group)}, {len(location_group)}, {len(type_group)}\n")
            print(group)
            missing_labels += 1
            print("\n")
    print(f"Skipped {missing_labels} groups due to inconsistent lengths.")
    return categories, indoors_or_outdoors, man_made_or_natural



def convert_annotation_to_int(annotation):
    category = -1
    location_map = {"indoors": 0, "outdoors": 1}
    type_map = {"man-made": 0, "natural": 1}
    indoors_or_outdoors = -1
    man_made_or_natural = -1
    for element in annotation:
        if element in location_map:
            indoors_or_outdoors = location_map[element]
        elif element in type_map:
            man_made_or_natural = type_map[element]
        elif element in categories_to_num_9:
            category = categories_to_num_9[element]
    return category, indoors_or_outdoors, man_made_or_natural


def load_llm_annotations(filepaths):
    """
    Load and process LLM annotations from JSON files.
    """
    categories=[]
    indoors_or_outdoors_group=[]
    man_made_or_natural_group=[]
    for filepath in filepaths:
        category_temp=[]
        indoors_or_outdoors_temp=[]
        man_made_or_natural_temp=[]
        with open(filepath, 'r') as file:
            data = json.load(file)
            for item in data:
                caption = item['caption']
                annotation = item['annotation']
                category, indoors_or_outdoors,man_made_or_natural = convert_annotation_to_int(annotation)
                category_temp.append(category)
                indoors_or_outdoors_temp.append(indoors_or_outdoors)
                man_made_or_natural_temp.append(man_made_or_natural)
        categories.append(category_temp)
        indoors_or_outdoors_group.append(indoors_or_outdoors_temp)
        man_made_or_natural_group.append(man_made_or_natural_temp)
    categories = [list(row) for row in zip(*categories)]
    indoors_or_outdoors_group = [list(row) for row in zip(*indoors_or_outdoors_group)]
    man_made_or_natural_group = [list(row) for row in zip(*man_made_or_natural_group)]
    return categories, indoors_or_outdoors_group, man_made_or_natural_group

def fleiss_kappa_for_human_data(filepaths, group_by_column):
    categories, indoors_or_outdoors, man_made_or_natural = analyze_data(filepaths, group_by_column)
    print(f"Data shapes - Categories: {np.shape(categories)}, Indoors/Outdoors: {np.shape(indoors_or_outdoors)}, Man-made/Natural: {np.shape(man_made_or_natural)}")
    calculate_fleiss_kappas(categories, indoors_or_outdoors, man_made_or_natural)

def fleiss_kappa_for_llm_annotations(filepath):
    categories, indoors_or_outdoors_group, man_made_or_natural_group = load_llm_annotations(filepath)
    print("\n")
    print("printing shapes",np.array(categories).shape,np.array(indoors_or_outdoors_group).shape,np.array(man_made_or_natural_group).shape) 
    print("calculating LLM fleiss kappas - CAPTIONS")
    calculate_fleiss_kappas(categories, indoors_or_outdoors_group,man_made_or_natural_group)


if __name__ == "__main__":
 
    
    ###inputs
    captions_filepaths = ["../../data/results/captions/captions.csv"]
    images_filepaths = ["../../data/results/images/images.csv"]
    llm_annotations_filepaths = [
    "../../data/results/llm_annotations/deepseek_annotations.json",
    "../../data/results/llm_annotations/perplexity_annotations.json",
    "../../data/results/llm_annotations/claude_annotations.json",
    "../../data/results/llm_annotations/chatgpt_annotations.json"
    ]

    # Analyze human annotations for captions
    fleiss_kappa_for_human_data(captions_filepaths, 'Input.sentence')

    # Analyze human annotations for images
    fleiss_kappa_for_human_data(images_filepaths, 'Input.image_url')

    # Analyze LLM annotations
    fleiss_kappa_for_llm_annotations(llm_annotations_filepaths)
   
