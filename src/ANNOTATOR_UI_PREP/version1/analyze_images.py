import csv
from collections import defaultdict

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


def parse_image_name(name):
    return name.split("-")[0]


def analyze_annotations(file_path):
    annotations = defaultdict(list)
    total = 0
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_url = row.get("Input.image_url", "")
            if not image_url:
                continue  # Skip rows without image URL

            image_name = image_url.split("/")[-1]  # Extract filename from URL
            base_name = parse_image_name(image_name)

            category = next(
                (
                    str(i)
                    for i in range(10)
                    if row.get(f"Answer.category.{i}") == "true"
                ),
                "Unknown",
            )
            category = (
                category_mapping.get(int(category), "Unknown")
                if category != "Unknown"
                else "Unknown"
            )

            worker_id = row.get("WorkerId", "Unknown")
            worker_name = id_mapping.get(worker_id, worker_id)

            annotation = {
                "worker_id": worker_name,
                "image": image_name,
                "location": "indoors"
                if row.get("Answer.location.indoors") == "true"
                else "outdoors",
                "type": "man-made"
                if row.get("Answer.type.man-made") == "true"
                else "natural",
                "category": category,
            }

            annotations[base_name].append(annotation)
            total += 1

    print(f"\nTotal annotations: {total}")
    print()

    diff = 0
    for base_name, image_annotations in sorted(annotations.items()):
        # print(f"Image base: {base_name}")
        # for annotation in image_annotations:
        #     print(f"  Worker: {annotation['worker_id']}, image: {annotation['image']}, Location: {annotation['location']}, Type: {annotation['type']}, Category: {annotation['category']}")

        differences = compare_annotations(image_annotations)
        if differences:
            diff += 1
            print(f"Image base: {base_name}")
            print("Differences:")
            for annotation in image_annotations:
                print(
                    f"  Worker: {annotation['worker_id']}, image: {annotation['image']}, Location: {annotation['location']}, Type: {annotation['type']}, Category: {annotation['category']}"
                )
            print(differences)
        print()
    print("size of difference is ", diff)
    print("size of total is ", total / 3)


def compare_annotations(annotations):
    if len(annotations) <= 1:
        return None

    differences = []
    first = annotations[0]
    for i, annotation in enumerate(annotations[1:], 1):
        diff = {}
        for key in ["location", "type", "category"]:
            if annotation[key] != first[key]:
                diff[
                    key
                ] = f"{first['worker_id']} ({first[key]}) -> {annotation['worker_id']} ({annotation[key]})"
        if diff:
            differences.append(f"Difference in annotation {i}:")
            for key, value in diff.items():
                differences.append(f"  {key}: {value}")

    return "\n".join(differences) if differences else None


# Usage
input_file = "images.csv"
analyze_annotations(input_file)
