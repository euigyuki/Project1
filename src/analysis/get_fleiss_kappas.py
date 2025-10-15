import pandas as pd
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
import numpy as np
from src.helper.helper_functions import WORKERS,categories_to_num_9,LLMS
from dataclasses import dataclass
from pathlib import Path


def calculate_fleiss_kappa(data):
    """
    Calculates Fleiss' Kappa for the provided data.
    """
    aggregated_data, _ = aggregate_raters(data)
    kappa = fleiss_kappa(aggregated_data)
    return kappa



def process_group(group,workers_or_llms):
    """
    Processes a group of data to extract category, location, and type annotations.
    """
    category_group = []
    indoors_or_outdoors_group = []
    man_made_or_natural_group = []
    for _, row in group.iterrows():
        if row['WorkerId'] not in workers_or_llms:
            continue
        parts=str.split(row['Answer.taskAnswers'], "/")
        category = parts[2]
        indoors_or_outdoors = parts[0]
        man_made_or_natural = parts[1]
        category_group.append(category)
        indoors_or_outdoors_group.append(indoors_or_outdoors)
        man_made_or_natural_group.append(man_made_or_natural)
    return category_group, indoors_or_outdoors_group, man_made_or_natural_group

def calculate_fleiss_kappas(categories, indoors_or_outdoors, man_made_or_natural):
    print(f"Fleiss' Kappa - categories: {calculate_fleiss_kappa(categories)}")
    print(f"Fleiss' Kappa - indoors or outdoors: {calculate_fleiss_kappa(indoors_or_outdoors)}")
    print(f"Fleiss' Kappa - man-made or natural: {calculate_fleiss_kappa(man_made_or_natural)}")

def analyze_data(filepaths, group_by_column,workers_or_llms):
    combined_data = pd.read_csv(filepaths)
    selected_columns = combined_data[['WorkerId', group_by_column, 'Answer.taskAnswers']]
    grouped = selected_columns.groupby(group_by_column)
    print(f"Number of groups: {len(grouped)}")
    categories, indoors_or_outdoors, man_made_or_natural = [], [], []
    missing_labels = 0
    for name, group in grouped:
        category_group, location_group, type_group = process_group(group,workers_or_llms)
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
    print(f"Skipped {missing_labels} groups due to inconsistent lengths.# of missing labels{missing_labels}")
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

def load_captions_filepaths(path_config):
    return path_config.captions_filepaths

def load_images_filepaths(path_config):
    return path_config.images_filepaths

def load_llm_annotations(path_config):
    return path_config.llm_annotations_filepaths

def fleiss_kappa_for_all_three(path_config,load_func, group_by_column,workers_or_llms):
    filepaths = load_func(path_config)
    categories, indoors_or_outdoors, man_made_or_natural = analyze_data(filepaths, group_by_column,workers_or_llms)
    print(f"Data shapes - Categories: {np.shape(categories)}, Indoors/Outdoors: {np.shape(indoors_or_outdoors)}, Man-made/Natural: {np.shape(man_made_or_natural)}")
    calculate_fleiss_kappas(categories, indoors_or_outdoors, man_made_or_natural)


@dataclass
class PathConfig:
    captions_filepaths: Path
    images_filepaths: Path
    llm_annotations_filepaths: Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

if __name__ == "__main__":
    path_config = PathConfig(
        ## Input paths
        captions_filepaths=DATA_DIR / "results" / "captions_annotated_by_humans"/ "captions_annotated_by_humans.csv",
        images_filepaths=DATA_DIR / "results" / "images_annotated_by_humans"/"images_annotated_by_humans.csv",
        llm_annotations_filepaths=DATA_DIR / "results" / "llm_annotations" / "captions_annotated_by_llms.csv"
    )
    
   
    # Analyze human annotations for captions
    fleiss_kappa_for_all_three(path_config,load_captions_filepaths,'Input.sentence',WORKERS)

    # Analyze human annotations for images
    fleiss_kappa_for_all_three(path_config,load_images_filepaths, 'Input.image_url',WORKERS)

    # Analyze LLM annotations
    fleiss_kappa_for_all_three(path_config,load_llm_annotations,"Input.sentence",LLMS)
   
