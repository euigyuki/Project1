import pandas as pd
import numpy as np
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