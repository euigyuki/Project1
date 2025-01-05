import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from typing import List, Tuple
import re

import os

os.environ["HF_TOKEN"] = "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"

# Load LLaMA model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Choose the appropriate model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)


def load_captions(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f]


def annotate_caption(
    caption: str, hierarchy: List[Tuple[str, str, str]]
) -> Tuple[str, str, str]:
    prompt = f"""Classify the following caption according to the given hierarchy.
Return only the matching tuple from the hierarchy, nothing else.

Caption: "{caption}"

Hierarchy:
{hierarchy}

Classification:"""

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(
        model.device
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=50, num_return_sequences=1, temperature=0.7
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
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
    prompt = f"""Explain why the following annotation was chosen for the given caption.

Caption: "{caption}"
Annotation: {annotation}

Explanation:"""

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(
        model.device
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=100, num_return_sequences=1, temperature=0.7
        )
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    explanation_match = re.search(r"Explanation:\s*(.*)", explanation, re.DOTALL)
    if explanation_match:
        return explanation_match.group(1).strip()
    else:
        return explanation


def main():
    captions = load_captions("captions.txt")  # Update the filename if needed
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
        print(f"Processing caption {i + 1} of {len(captions)}")
        annotation = annotate_caption(caption, all_combinations)
        explanation = explain_annotation(caption, annotation)
        results.append(
            {"caption": caption, "annotation": annotation, "explanation": explanation}
        )

    with open("llama_annotations_with_explanations.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        "Processing complete. Results saved to llama_annotations_with_explanations.json"
    )


if __name__ == "__main__":
    main()
