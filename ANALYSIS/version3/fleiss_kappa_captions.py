from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import statsmodels.stats.inter_rater as irr
from fleiss_kappa import (
    get_category_labels,
    get_true_labels,
    load_config,
    read_answers,
    pretty_print_matrix,
)


def calculate_matches(list1, list2):
    # Ensure the lists are of the same length
    min_length = min(len(list1), len(list2))

    # Count matches
    matches = sum(a == b for a, b in zip(list1[:min_length], list2[:min_length]))

    # Calculate percentage
    percentage = (matches / min_length) * 100

    return matches, percentage


def calculate_fleiss_kappa(labels_func, param1, param2=None):
    config = load_config("config.yaml")
    worker_ids = [config["workers"][f"worker_id_{i}"] for i in range(1, 3)]
    df = pd.read_csv(f"captions{config['version']}.csv")

    if param2 is not None:
        labels = [
            labels_func(read_answers(df, worker_id), param1, param2)
            for worker_id in worker_ids
        ]
    else:
        labels = [
            labels_func(read_answers(df, worker_id), param1) for worker_id in worker_ids
        ]

    min_length = min(len(l) for l in labels)
    num_categories = len(set(label for l in labels for label in l))
    print(f"Number of categories: {num_categories}")
    print("Unique categories:", set(label for l in labels for label in l))
    ratings_matrix = np.zeros((min_length, num_categories))
    labels[1].pop(6)
    # print("labels ",labels[1].pop(6))

    for i in range(min_length):
        for j, label in enumerate(labels):
            ratings_matrix[i][label[i]] += 1

    kappa = irr.fleiss_kappa(ratings_matrix)
    annotators = ["Derrick", "Ken", "Sonny"]
    for i, l in enumerate(labels):
        print(f"\nCategory Labels {annotators[i]}:")
        print(l)
        print(f"length of labels: {len(l)}")

    matches, percentage = calculate_matches(labels[0], labels[1])
    print(f"Number of matches: {matches}")
    print(f"Percentage of matches: {percentage:.2f}%")

    # print("\nRatings Matrix:")
    # pretty_print_matrix(ratings_matrix)
    if param2:
        print(f"\nFLEISS KAPPA FOR {param2.upper()}: {kappa:.3f}")
    else:
        print(f"\nFLEISS KAPPA: {param1.upper()}:{kappa:.3f}")


def main():
    # Calculate Fleiss' Kappa for man-made/natural
    calculate_fleiss_kappa(get_true_labels, "type", "man-made")

    # Calculate Fleiss' Kappa for indoors/outdoors
    calculate_fleiss_kappa(get_true_labels, "location", "outdoors")

    # Calculate Fleiss' Kappa for categories
    calculate_fleiss_kappa(get_category_labels, "category")


if __name__ == "__main__":
    main()
