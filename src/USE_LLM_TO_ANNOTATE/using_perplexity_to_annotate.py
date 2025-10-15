import requests
import json
from typing import List, Tuple
import re
from utils import load_first_column
import os

# Retrieve the API key
api_key = os.getenv("perplexity_api_key")
print("api key is", api_key)

def get_perplexity_client(api_key: str):
    """Initialize the Perplexity client."""
    return {
        "headers": {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        "url": "https://api.perplexity.ai/chat/completions",
    }


def load_captions(file_path: str) -> List[str]:
    """Load captions from a text file."""
    with open(file_path, "r") as f:
        return [line.strip() for line in f]


def annotate_caption(
    caption: str, hierarchy: list, client: dict
) -> Tuple[str, str, str]:
    """Classify a caption according to the given hierarchy using Perplexity."""
    prompt = f"""Classify the following caption according to the given hierarchy.

Caption: '{caption}'

Hierarchy: {hierarchy}

Instructions:
1. Choose exactly one classification from the hierarchy for each of the three levels:
   - First, classify where the caption is happening: indoors or outdoors.
   - Second, classify whether it involves man-made or natural elements.
   - Third, select the most fitting category based on the activity or context in the caption (e.g., work, recreation, etc.)
2. Provide a full classification in the format: ('location', 'nature', 'activity')
3. Only include the classification tuple and no other text.
4. Do not omit any category, even if it is less obvious.

Classification:"""

    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that classifies captions.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
    }

    try:
        response = requests.post(client["url"], headers=client["headers"], json=payload)
        response.raise_for_status()
        result = response.json()
        print("API Response:", json.dumps(result, indent=2))

        content = result["choices"][0]["message"]["content"]

        match = re.search(r"\(.*?\)", content)
        if match:
            tuple_str = match.group(0)
            try:
                annotation = eval(tuple_str)
                if isinstance(annotation, tuple) and len(annotation) == 3:
                    return annotation
            except:
                print(f"Failed to parse annotation: {tuple_str}")
        else:
            print(f"No tuple found in content: {content}")
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        print(
            "Response content:",
            e.response.text if e.response else "No response content",
        )
    except (KeyError, IndexError) as e:
        print(f"Failed to extract content from API response: {e}")
    except json.JSONDecodeError as e:
        print(f"Failed to decode API response: {e}")
        print("Raw response:", response.text)

    return ("unknown", "unknown", "unknown")


def explain_annotation(
    caption: str, annotation: Tuple[str, str, str], client: dict
) -> str:
    """Generate an explanation for the annotation using Perplexity."""
    prompt = f"""Explain the following annotation for the given caption.

Caption: '{caption}'
Annotation: {annotation}

Instructions:
1. For each category in the annotation, provide a brief justification for why that category was chosen.
2. Follow this format:
   "The caption is classified as [location] because [reason], involves [nature] elements because [reason], and shows [activity] because [reason]."
3. Ensure the explanation is clear, concise, and directly relevant to the caption.
4. Only provide the explanation, no additional text.

Explanation:"""

    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that explains caption classifications.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
    }

    try:
        response = requests.post(client["url"], headers=client["headers"], json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        print(
            "Response content:",
            e.response.text if e.response else "No response content",
        )
    except (KeyError, IndexError) as e:
        print(f"Failed to extract content from API response: {e}")
    except json.JSONDecodeError as e:
        print(f"Failed to decode API response: {e}")
        print("Raw response:", response.text)

    return "Failed to generate explanation"


def main():
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

    try:
        client = get_perplexity_client(api_key)
        
        file_path = "../../data/results/total_captions.csv"
        captions = load_first_column(file_path)

        results = []

        for i, caption in enumerate(captions):
            print(f"Processing caption {i+1} of {len(captions)}: {caption}")

            annotation = annotate_caption(caption, all_combinations, client)
            print(f"Annotation: {annotation}")

            explanation = explain_annotation(caption, annotation, client)
            print(f"Explanation: {explanation}")
            print("-" * 80)

            results.append(
                {
                    "caption": caption,
                    "annotation": annotation,
                    "explanation": explanation,
                }
            )

        with open("../../data/results/perplexity_annotations.json", "w") as f:
            json.dump(results, f, indent=2)

        print("Processing complete. Results saved to perplexity_annotations.json")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
