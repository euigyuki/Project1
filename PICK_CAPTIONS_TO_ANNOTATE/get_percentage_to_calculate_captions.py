import csv
from collections import defaultdict

# Define the categories and their associated words
categories = {
    "Indoor/manmade/Transport-Urban": ["subway"],
    "Indoor/manmade/Restaurant": ["restaurant", "bar"],
    "Indoor/manmade/Recreation": ["pool"],
    "Indoor/manmade/Domestic": ["room", "kitchen", "floor", "window", "chair"],
    "Indoor/manmade/Work-Education": [],
    "Indoor/manmade/Other-Unclear": ["table", "stage", "metal", "board"],
    "Outdoor/man-made/Transport-Urban": [
        "street",
        "sidewalk",
        "city",
        "car",
        "road",
        "construction",
        "train",
        "track",
        "truck",
        "bus",
        "bridge",
        "pole",
    ],
    "Outdoor/man-made/Restaurant": [],
    "Outdoor/man-made/Recreation": ["park", "bench", "skateboard", "fountain"],
    "Outdoor/man-made/Domestic": ["house"],
    "Outdoor/man-made/Work-Education": [],
    "Outdoor/man-made/Other-Unclear": [
        "building",
        "sign",
        "market",
        "fence",
        "brick",
        "light",
        "shop",
    ],
    "Outdoor/Natural/Body-of-Water": [
        "water",
        "ocean",
        "lake",
        "river",
        "wave",
        "beach",
    ],
    "Outdoor/Natural/Field-Forest": [
        "field",
        "grass",
        "tree",
        "trees",
        "grassy",
        "flowers",
    ],
    "Outdoor/Natural/Mountain": ["mountain", "hill"],
    "Outdoor/Natural/Other-Unclear": [
        "snow",
        "rock",
        "dirt",
        "ground",
        "sand",
        "ice",
        "rocks",
        "stone",
        "snowy",
    ],
    "Outdoor/Natural/Other": [
        "sitting",
        "standing",
        "playing",
        "outside",
        "near",
        "play",
        "game",
        "head",
        "baseball",
        "race",
        "boat",
        "suit",
        "football",
        "band",
        "eating",
        "haired",
        "horse",
    ],
}


def read_captions(file_path):
    captions = []
    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row if present
        for row in reader:
            captions.append(row[0].lower())  # Assuming captions are in the first column
    print(f"Total captions: {len(captions)}")
    return captions


def calculate_percentages(captions, categories):
    total_captions = len(captions)
    results = defaultdict(lambda: defaultdict(int))
    caption_counted = [False] * total_captions
    for idx, caption in enumerate(captions):
        words = set(caption.split())
        for category, category_words in categories.items():
            for word in category_words:
                if word in words:
                    results[category][word] += 1
                    caption_counted[idx] = True
                    break  # Count each caption only once per category
            if caption_counted[idx]:
                break  # Move to next caption if this one has been counted
    percentages = {}
    total_percentage = 0
    for category, word_counts in results.items():
        category_count = sum(word_counts.values())
        category_percentage = category_count / total_captions * 100
        total_percentage += category_percentage
        percentages[category] = {
            "percentage": category_percentage,
            "word_percentages": {
                word: (count / total_captions * 100)
                for word, count in word_counts.items()
            },
        }
    uncategorized_count = caption_counted.count(False)
    uncategorized_percentage = uncategorized_count / total_captions * 100
    total_percentage += uncategorized_percentage
    percentages["Uncategorized"] = {
        "percentage": uncategorized_percentage,
        "word_percentages": {},
    }

    print(f"Total percentage: {total_percentage:.2f}%")
    if (
        abs(total_percentage - 100) > 0.01
    ):  # Allow for small floating-point discrepancies
        print("Warning: Total percentage does not equal 100%.")

    return percentages


def print_results(percentages):
    for category, data in percentages.items():
        print(f"\n{category}:")
        print(f"Overall percentage: {data['percentage']:.2f}%")
        if category != "Uncategorized":
            print("Word percentages:")
            for word, percentage in data["word_percentages"].items():
                print(f"  {word}: {percentage:.2f}%")


def main():
    """input and output file paths"""
    file_path = "results.csv"
    """input and output file paths"""

    captions = read_captions(file_path)
    percentages = calculate_percentages(captions, categories)
    print_results(percentages)


if __name__ == "__main__":
    main()


