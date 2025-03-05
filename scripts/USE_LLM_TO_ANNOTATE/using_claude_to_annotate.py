import anthropic
import json
from typing import List, Tuple
import configparser
from utils import load_first_column

# Initialize the parser
config = configparser.ConfigParser()

# Read the config file
config.read('config.cfg')

# Retrieve the API key
api_key = config.get('anthropic', 'api_key')


def annotate_caption(caption: str, hierarchy: List[Tuple[str, str, str]], client: anthropic.Client) -> Tuple[str, str, str]:
    messages = [
        {"role": "user", "content": f"Given the following caption, classify it according to the hierarchy below. Return only the matching tuple from the hierarchy, nothing else.\n\nCaption: \"{caption}\"\n\nHierarchy:\n{hierarchy}\n\nClassification:"}
    ]
    
    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            messages=messages,
            max_tokens=50,
            temperature=0,
        )
        # Access the content attribute directly
        result = response.content[0].text.strip()
        return eval(result)
    except Exception as e:
        print(f"Error during annotation: {e}")
        return ("error", "error", "error")

def explain_annotation(caption: str, annotation: Tuple[str, str, str], client: anthropic.Client) -> str:
    messages = [
        {"role": "user", "content": f"Given the following caption and its annotation, explain why this annotation was chosen.\n\nCaption: \"{caption}\"\nAnnotation: {annotation}\n\nExplanation:"}
    ]
    
    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            messages=messages,
            max_tokens=100,
            temperature=0,
        )
        # Access the content attribute directly
        return response.content[0].text.strip()
    except Exception as e:
        print(f"Error during explanation: {e}")
        return "An error occurred during explanation."


def main():
    client = anthropic.Client(api_key=api_key)
    file_path = "../../data/results/total_captions.csv"
    captions = load_first_column(file_path)
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
        annotation = annotate_caption(caption, all_combinations, client)
        explanation = explain_annotation(caption, annotation, client)
        results.append(
            {"caption": caption, "annotation": annotation, "explanation": explanation}
        )

    # Save results to a JSON file
    with open("../../data/results/claude_annotations_with_explanations.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        "Processing complete. Results saved to claude_annotations_with_explanations.json"
    )

if __name__ == "__main__":
    main()
