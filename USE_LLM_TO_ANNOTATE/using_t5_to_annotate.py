import json
from typing import List, Tuple
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

# model_name = "t5-large"
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def load_captions(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f]


def annotate_caption(caption, hierarchy):
    prompt = f"""Classify the following caption according to the given hierarchy.

Caption: '{caption}'

Hierarchy: {hierarchy}


Instructions:
1. Choose exactly one classification from the hierarchy for each of the three levels:
   - First, classify where the caption is happening: indoors or outdoors.
   - Second, classify whether it involves man-made or natural elements.
   - Third, select the most fitting category based on the activity or context in the caption (e.g., work, recreation, etc.)
2. Provide a full classification in the format: ('location', 'nature', 'activity')
3. Only include the classification and no other text.
4. Do not omit any category, even if it is less obvious.

Classification:"""

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        num_return_sequences=1,
        temperature=0.5,
        do_sample=False,  # Set to False to avoid random sampling
        top_k=50,
        top_p=0.95,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    print(f"Raw model output for caption '{caption}': {result}")

    # More flexible regex pattern to extract categories
    match = re.search(r"\(.*?\)", result)
    if match:
        tuple_str = match.group(0)
        try:
            annotation = eval(tuple_str)
            if isinstance(annotation, tuple) and len(annotation) == 3:
                return annotation
        except:
            pass
    print(f"Error parsing result: {result}")
    return ("unknown", "unknown", "unknown")


def explain_annotation(caption: str, annotation: Tuple[str, str, str]) -> str:
    prompt = f"""Explain the following annotation for the given caption.

Caption: '{caption}'
Annotation: {annotation}

Instructions:
   1. For each category in the annotation, provide a brief justification for why that category was chosen.
   2. Follow this example format:
      - "Annotation: ('outdoors', 'man-made', 'work_education') -> Justification: This caption refers to people working outside, using a man-made system (pulley)."
   3. Ensure the explanation is clear, concise, and directly relevant to the caption.



Your explanation:"""

    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        num_return_sequences=1,
        temperature=0.5,
        do_sample=False,  # Set to False to avoid random sampling
        top_k=50,
        top_p=0.95,
    )
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    explanation_match = re.search(
        r"Detailed Explanation:\s*(.*)", explanation, re.DOTALL
    )
    if explanation_match:
        return explanation_match.group(1).strip()
    else:
        return explanation


def main():
    captions = load_captions("captions1.txt")
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
    for i, caption in enumerate(captions):
        print(f"Processing caption {i} of {len(captions)}")
        annotation = annotate_caption(caption, all_combinations)
        print("Annotation:", annotation)
        print("#" * 50)
        explanation = explain_annotation(caption, annotation)
        print("Explanation:", explanation)
        print("#" * 50)
        results.append(
            {"caption": caption, "annotation": annotation, "explanation": explanation}
        )

    with open("t5_annotations_with_explanations.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Processing complete. Results saved to t5_annotations_with_explanations.json")


if __name__ == "__main__":
    main()
