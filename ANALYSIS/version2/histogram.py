import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
from confusion_matrix import get_true_labels
from sklearn.metrics import confusion_matrix
from fleiss_kappa import load_config, get_category_labels

config = load_config("config.yaml")


def histogram(path, confusion_matrix, images_or_captions, categories, element):
    # Generate a histogram
    plt.figure(figsize=(8, 6))
    values = [
        confusion_matrix[0, 0],
        confusion_matrix[0, 1],
        confusion_matrix[1, 0],
        confusion_matrix[1, 1],
    ]
    bars = plt.bar(
        categories, values, color=["skyblue", "lightcoral", "lightcoral", "skyblue"]
    )

    # Annotate bars with their values
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            "{}".format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # Offset text by 3 points above the bar
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.title("Histogram of Confusion Matrix Elements")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.savefig(f"{path}/{images_or_captions}_{element}_confusion_matrix_histogram.png")


def load_cm(df, param1, param2):
    worker_id_1 = config["workers"]["worker_id_1"]
    worker_id_2 = config["workers"]["worker_id_2"]
    # worker_id_3 = config['workers']['worker_id_3']
    answers_1 = df[df["WorkerId"] == worker_id_1]["Answer.taskAnswers"]
    answers_2 = df[df["WorkerId"] == worker_id_2]["Answer.taskAnswers"]
    # answers_3 = df[df['WorkerId'] == worker_id_3]['Answer.taskAnswers']
    derrick = get_true_labels(answers_1, param1, param2)
    ken = get_true_labels(answers_2, param1, param2)
    cm = confusion_matrix(derrick, ken)
    return cm


def manmade(path, df, images_or_captions):
    param1 = "type"
    param2 = "man-made"
    cm = load_cm(df, param1, param2)
    categories = ["True man-made", "False natural", "False man-made", "True natural"]
    histogram(path, cm, images_or_captions, categories, element="manmade")


def outdoors(path, df, images_or_captions):
    param1 = "location"
    param2 = "outdoors"
    cm = load_cm(df, param1, param2)
    categories = ["True Indoors", "False Outdoors", "False Indoors", "True Outdoors"]
    histogram(path, cm, images_or_captions, categories, element="outdoors")


def histogram_categories(path, cm, images_or_captions, categories, element):
    plt.figure(figsize=(12, 10))
    # Flatten the confusion matrix for plotting
    values = np.sum(cm, axis=0)  # Sum predictions for each actual category
    num_categories = len(categories)
    bars = plt.bar(
        range(num_categories),
        values[:num_categories],
        color=sns.color_palette("husl", num_categories),
    )

    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            "{}".format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.title(f"Histogram of {element} Confusion Matrix Elements")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(ticks=range(num_categories), labels=categories, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{path}/{images_or_captions}_{element}_confusion_matrix_histogram.png")


def categories(path, df, images_or_captions):
    param1 = "category"
    worker_id_1 = config["workers"]["worker_id_1"]
    worker_id_2 = config["workers"]["worker_id_2"]

    answers_1 = df[df["WorkerId"] == worker_id_1]["Answer.taskAnswers"]
    answers_2 = df[df["WorkerId"] == worker_id_2]["Answer.taskAnswers"]

    derrick = get_category_labels(answers_1, param1)
    ken = get_category_labels(answers_2, param1)

    cm = confusion_matrix(derrick, ken, labels=range(len(config["category_mapping"])))

    category_labels = [
        config["category_mapping"][i] for i in range(len(config["category_mapping"]))
    ]

    histogram_categories(
        path, cm, images_or_captions, category_labels, element="categories"
    )


def main():
    version = config["version"]
    images_or_captions = "images"
    # category_mapping = config['category_mapping']
    # id_mapping = config['id_mapping']
    # name_to_id = config['name_to_id']

    filename = f"{images_or_captions}{version}.csv"
    df = pd.read_csv(filename)
    path = "histogram"

    manmade(path, df, images_or_captions)
    outdoors(path, df, images_or_captions)
    categories(path, df, images_or_captions)


if __name__ == "__main__":
    main()
