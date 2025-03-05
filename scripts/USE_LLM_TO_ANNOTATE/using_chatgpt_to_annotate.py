import openai
import json
from typing import List, Tuple
import pandas as pd
from utils import load_first_column
import os


api_key = os.getenv("openai_api_key")




def annotate_caption(caption: str, hierarchy: List[Tuple[str, str, str]]) -> Tuple[str, str, str]:
    prompt = f"""
    Given the following caption, classify it according to the hierarchy below.
    Return only the matching tuple from the hierarchy, nothing else.

    Caption: "{caption}"

    Hierarchy:
    {hierarchy}

    Classification:
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that classifies captions according to a given hierarchy.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
        )
        result = response.choices[0].message.content.strip()
        return eval(result)
    except Exception as e:
        print(f"Error during annotation: {e}")
        return ("error", "error", "error")

def explain_annotation(caption: str, annotation: Tuple[str, str, str]) -> str:
    prompt = f"""
    Given the following caption and its annotation, explain why this annotation was chosen.

    Caption: "{caption}"
    Annotation: {annotation}

    Explanation:
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that explains caption classifications.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during explanation: {e}")
        return "An error occurred during explanation."

def main():
    file_path = "../../data/results/total_captions.csv"
    captions = load_first_column(file_path)
    print(f"length of captions: {len(captions)}")
    all_combinations = [
        ("indoors", "man-made", "work_education"),
        ("indoors", "man-made", "domestic"),
        ("indoors", "man-made", "recreation"),
        ("indoors", "man-made", "restaurant"),
        ("indoors", "man-made", "transportation_urban"),
        ("indoors", "man-made", "other_unclear"),
        ("outdoors", "man-made", "work_education"),
        ("outdoors", "man-made", "domestic"),
        ("outdoors", "man-made", "recreation"),
        ("outdoors", "man-made", "restaurant"),
        ("outdoors", "man-made", "transportation_urban"),
        ("outdoors", "man-made", "other_unclear"),
        ("outdoors", "natural", "field_forest"),
        ("outdoors", "natural", "body_of_water"),
        ("outdoors", "natural", "mountain"),
        ("outdoors", "natural", "other_unclear"),
    ]

    results = []
    for i, caption in enumerate(captions, 1):
        print(f"Processing caption {i} of {len(captions)}")
        annotation = annotate_caption(caption, all_combinations)
        explanation = explain_annotation(caption, annotation)
        results.append(
            {"caption": caption, "annotation": annotation, "explanation": explanation}
        )

    # Save results to a JSON file
    with open("../../data/results/chatgpt_annotations_with_explanations.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Processing complete. Results saved")

if __name__ == "__main__":
    main()
