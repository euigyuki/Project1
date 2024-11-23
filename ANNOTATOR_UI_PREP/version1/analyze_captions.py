import csv
import json
from collections import defaultdict, Counter

category_mapping = {
    0: "Transportation/Urban",
    1: "Restaurant",
    2: "Recreation",
    3: "Domestic",
    4: "Work/Education",
    5: "Other/Unclear (Man-made)",
    6: "Body of Water",
    7: "Field/Forest",
    8: "Mountain",
    9: "Other/Unclear (Natural)",
}

id_mapping = {
    "A17EZEAMF37MGQ": "Derrick",
    "AO2A58R9LMCKC": "Ken",
    "A6I9Z5XQ933N4": "Sonny",
    "A3M2XET3KDDS9S": "Unknown",
}


def analyze_annotations(input_file):
    annotations = defaultdict(list)
    worker_counts = Counter()

    with open(input_file, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            hit_id = row.get("HITId", "Unknown")
            worker_id = row.get("WorkerId", "Unknown")

            # Parse the Answer.taskAnswers JSON
            try:
                task_answers = json.loads(row.get("Answer.taskAnswers", "[]"))
                if task_answers:
                    answer = task_answers[0]
                    category_index = next(
                        (int(k) for k, v in answer.get("category", {}).items() if v),
                        None,
                    )
                    category = (
                        category_mapping.get(category_index, "Unknown")
                        if category_index is not None
                        else "Unknown"
                    )
                    location = (
                        "indoors"
                        if answer.get("location", {}).get("indoors")
                        else "outdoors"
                    )
                    type_ = (
                        "man-made"
                        if answer.get("type", {}).get("man-made")
                        else "natural"
                    )
                else:
                    category, location, type_ = "Unknown", "Unknown", "Unknown"
            except json.JSONDecodeError:
                category, location, type_ = "Unknown", "Unknown", "Unknown"

            annotation = {
                "worker_id": id_mapping.get(worker_id, worker_id),
                "sentence": row.get("Input.sentence", "Unknown"),
                "location": location,
                "type": type_,
                "category": category,
            }

            annotations[hit_id].append(annotation)
            worker_counts[worker_id] += 1

    print_differences(annotations)
    print_worker_stats(worker_counts)


def print_differences(annotations):
    difference_count = 1
    print("Annotations with differences:")
    for hit_id, hit_annotations in annotations.items():
        if len(hit_annotations) > 1:
            categories = set(a["category"] for a in hit_annotations)
            locations = set(a["location"] for a in hit_annotations)
            types = set(a["type"] for a in hit_annotations)

            if len(categories) > 1 or len(locations) > 1 or len(types) > 1:
                print(f"{difference_count} Sentence: {hit_annotations[0]['sentence']}")
                difference_count += 1
                for annotation in hit_annotations:
                    print(
                        f"Worker: {annotation['worker_id']}, Category: {annotation['category']}, "
                        f"Location: {annotation['location']}, Type: {annotation['type']}"
                    )
                print()
    print(f"Total annotations with differences: {difference_count}")


def print_worker_stats(worker_counts):
    print("\nWorker annotation counts:")
    for worker_id, count in worker_counts.items():
        worker_name = id_mapping.get(worker_id, worker_id)
        print(f"{worker_name}: {count}")


# Usage
input_file = "captions.csv"
analyze_annotations(input_file)
