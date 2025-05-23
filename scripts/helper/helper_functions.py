import pandas as pd
import numpy as np
import re,json

categories_to_num_16 = {
    "outdoors/man-made/transportation_urban":0,
    "outdoors/man-made/recreation":1,  
    "indoors/man-made/recreation":2,  
    "outdoors/natural/body_of_water":3, 
    "outdoors/natural/field_forest":4,  
    "indoors/man-made/domestic":5, 
    "indoors/man-made/work_education":6, 
    "outdoors/man-made/other_unclear":7, 
    "outdoors/man-made/domestic":8, 
    "outdoors/natural/mountain":9,
    "outdoors/man-made/work_education":10, 
    "indoors/man-made/other_unclear":11, 
    "indoors/man-made/restaurant":12, 
    "indoors/man-made/transportation_urban":13, 
    "outdoors/natural/other_unclear":14,
    "outdoors/man-made/restaurant":15
}
num_16_to_category = {v: k for k, v in categories_to_num_16.items()}

categories_to_num_9 = {
    "transportation_urban":0,
    "restaurant":1,
    "recreation":2,
    "domestic":3,
    "work_education":4,
    "other_unclear":5,
    "body_of_water":6,
    "field_forest":7,
    "mountain":8,
    "other_unclear":9
}
nums9_to_categories = {
    0: "transportation_urban",
    1: "restaurant",
    2: "recreation",
    3: "domestic",
    4: "work_education",
    5: "other_unclear",
    6: "body_of_water",
    7: "field_forest",
    8: "mountain",
    9: "other_unclear"
}

WORKERS = ["A17EZEAMF37MGQ",#derrick
"A176JUTGNWG7QJ",#sohini
"A2SMHEGRLML092",#Lindsay
"A2ZY94PZ5CVH0" #matt
]

LLMS=[
    "chatgpt",
    "claude",
    "deepseek",
    "perplexity"
]
VLMS=[
    "flux",
    "dalle",
    "midjourney"
]
ENTITY_TO_WORKERS = {
    "human": WORKERS,
    "llms": LLMS,
    "vlms": VLMS
}


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
            total = AnnotationProcessor.process_human_annotation(row['Answer.taskAnswers'])
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
                    total = AnnotationProcessor.process_llm_annotation(item['annotation'])
                    if total == "outdoors/natural/recreation":
                        total = "outdoors/man-made/recreation"
                    if caption in original_captions_set:
                        self.llm_annotations["original"].setdefault(caption, []).append(total)
                    else:
                        self.llm_annotations["finalized"].setdefault(caption, []).append(total)
    @staticmethod
    def process_llm_annotation(annotation):
        category = categories_to_num_9.get(annotation[2], -1)
        category = nums9_to_categories[category]
        location = "indoors" if 'indoors' in annotation else "outdoors"
        type_ = "man-made" if 'man-made' in annotation else "natural"
        total = f"{location}/{type_}/{category}"
        return total
    
    @staticmethod
    def _process_categories( categories_map):
        for key, value in categories_map.items():
            if value:
                return nums9_to_categories[int(key)]
    @staticmethod
    def process_human_annotation(input_json):
        answer_dict = json.loads(input_json)[0]
        category = AnnotationProcessor._process_categories(answer_dict['category'])
        location = "indoors" if bool_dict_to_int_list(answer_dict['location']) == 0 else "outdoors"
        type_ = "man-made" if bool_dict_to_int_list(answer_dict['type']) == 0 else "natural"
        total = f"{location}/{type_}/{category}"
        return total

def bool_dict_to_int_list(d):
    if len(d) == 10:
        for key,value in d.items():
            if value == True:
                return int(key)
    else:
        for key,value in d.items():
            if (key == "indoors" and value == True) or (key == "man-made" and value == True):
                return 0
            else:
                return 1

def strip_word(word):
    if isinstance(word, str):
        return re.sub(r'-\d+$', '', word)
    return word

def get_set_of(filepaths,value):
    combined_df = load_combined_df(filepaths)
    return set(combined_df[value])

def normalize_caption(caption):
    return caption.strip().strip('"').strip("'")  # Remove leading/trailing spaces and quotes

def clip_probs(probs, epsilon=1e-10, max_val=4):
    return np.clip(probs, epsilon, max_val)

def load_combined_df(filepaths):
    dataframes = [pd.read_csv(filepath) for filepath in filepaths]
    combined_df = pd.concat(dataframes, axis=0)
    return combined_df

def majority_vote_from_distribution(counts,adjudication=None):
    """
    Returns the majority-vote annotation string from a count vector.
    
    Args:
        counts (List[int]): A list of 16 integers representing annotation counts.
    
    Returns:
        str: The majority-vote annotation (e.g. 'outdoors/natural/body_of_water').
    """
    #print("counts",counts)
    if len(counts) != 16:
        raise ValueError("Input must be a list of length 16.")

    max_count = max(counts)
    if max_count == 0:
        return None  # or raise an error if needed
    #print("max_count",max_count)

    # Find all categories with max count (in case of tie)
    tied_indices = [i for i, count in enumerate(counts) if count == max_count]
    #print("tied_indices",tied_indices)
    if len(tied_indices) > 1:
        return num_16_to_category[adjudication]
    else:
        return num_16_to_category[tied_indices[0]]
        

# Function to calculate probability distribution
def calculate_probability_distribution(annotations):
    total = [0]*16
    for annotation in annotations:
        index = categories_to_num_16[annotation]
        total[index] += 1
    return total