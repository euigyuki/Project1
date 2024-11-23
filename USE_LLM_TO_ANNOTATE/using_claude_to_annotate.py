import anthropic
import json
from typing import List, Tuple

# Set your Anthropic API key
ANTHROPIC_API_KEY = "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"


def load_captions(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f]


def annotate_caption(
    caption: str, hierarchy: List[Tuple[str, str, str]], client: anthropic.Anthropic
) -> Tuple[str, str, str]:
    prompt = f"""
    Given the following caption, classify it according to the hierarchy below. 
    Return only the matching tuple from the hierarchy, nothing else.

    Caption: "{caption}"

    Hierarchy:
    {hierarchy}

    Classification:
    """

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=50,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    result = message.content[0].text.strip()
    # Convert string tuple to actual tuple
    return eval(result)


def explain_annotation(
    caption: str, annotation: Tuple[str, str, str], client: anthropic.Anthropic
) -> str:
    prompt = f"""
    Given the following caption and its annotation, explain why this annotation was chosen.

    Caption: "{caption}"
    Annotation: {annotation}

    Explanation:
    """

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text.strip()


def main():
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

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
    with open("claude_annotations_with_explanations.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        "Processing complete. Results saved to claude_annotations_with_explanations.json"
    )


if __name__ == "__main__":
    main()
