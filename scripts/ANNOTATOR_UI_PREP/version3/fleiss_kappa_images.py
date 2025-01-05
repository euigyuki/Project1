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


def calculate_fleiss_kappa(labels_func, param1, param2=None):
    config = load_config("config.yaml")
    worker_ids = [config["workers"][f"worker_id_{i}"] for i in range(1, 3)]
    df = pd.read_csv(f"images{config['version']}.csv")
    print(df.head())

    labels = []
    for worker_id in worker_ids:
        worker_labels = read_answers(df, worker_id)
        if not worker_labels.empty:
            labels.append(
                labels_func(worker_labels, param1, param2)
                if param2
                else labels_func(worker_labels, param1)
            )
        else:
            print(f"Warning: No data found for worker ID {worker_id}")

    # Check if we have labels for both workers
    if len(labels) < 2 or any(len(l) == 0 for l in labels):
        print(
            "Error: Insufficient data for calculation (one or more workers have no labels)."
        )
        return

    # Continue with the existing steps to calculate Fleiss' Kappa if data is sufficient
    min_length = min(len(l) for l in labels)
    unique_labels = set(label for l in labels for label in l)
    num_categories = len(unique_labels)
    print(f"Number of categories: {num_categories}")
    print("Unique categories:", unique_labels)

    ratings_matrix = np.zeros((min_length, num_categories), dtype=int)

    for i in range(min_length):
        for j, label in enumerate(labels):
            category_index = list(unique_labels).index(label[i])
            ratings_matrix[i][category_index] += 1

    # Calculate Fleiss' Kappa
    try:
        kappa = irr.fleiss_kappa(ratings_matrix)
        if param2:
            print(f"\nFLEISS KAPPA FOR {param2.upper()}: {kappa:.3f}")
        else:
            print(f"\nFLEISS KAPPA: {param1.upper()}: {kappa:.3f}")
    except ValueError as e:
        print(f"Error calculating Fleiss' Kappa: {e}")


def main():
    # Calculate Fleiss' Kappa for man-made/natural
    calculate_fleiss_kappa(get_true_labels, "type", "man-made")

    # Calculate Fleiss' Kappa for indoors/outdoors
    calculate_fleiss_kappa(get_true_labels, "location", "outdoors")

    # Calculate Fleiss' Kappa for categories
    calculate_fleiss_kappa(get_category_labels, "category")


if __name__ == "__main__":
    main()
