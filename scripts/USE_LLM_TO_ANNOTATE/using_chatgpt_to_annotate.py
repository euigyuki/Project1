import openai
import json
from typing import List, Tuple
from openai import OpenAI

# Set your OpenAI API key


def load_captions(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f]


def annotate_caption(
    caption: str, hierarchy: List[Tuple[str, str, str]], client: OpenAI
) -> Tuple[str, str, str]:
    prompt = f"""
    Given the following caption, classify it according to the hierarchy below. 
    Return only the matching tuple from the hierarchy, nothing else.

    Caption: "{caption}"

    Hierarchy:
    {hierarchy}

    Classification:
    """

    response = client.chat.completions.create(
        model="gpt-4-0613",
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
    # Convert string tuple to actual tuple
    return eval(result)


def explain_annotation(
    caption: str, annotation: Tuple[str, str, str], client: OpenAI
) -> str:
    prompt = f"""
    Given the following caption and its annotation, explain why this annotation was chosen.

    Caption: "{caption}"
    Annotation: {annotation}

    Explanation:
    """

    response = client.chat.completions.create(
        model="gpt-4-0613",
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


def main():
    api_key = "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"  # Replace with your actual OpenAI API key
    client = OpenAI(api_key=api_key)

    captions = load_captions("captions.txt")
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
    with open("chatgpt_annotations_with_explanations.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Processing complete. Results saved to annotations_with_explanations.json")


if __name__ == "__main__":
    main()
