from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

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
name_to_id = {
    "Derrick": "A17EZEAMF37MGQ",
    "Ken": "AO2A58R9LMCKC",
    "Sonny": "A6I9Z5XQ933N4",
    "Unknown": "A3M2XET3KDDS9S",
}

worker_id_1 = name_to_id["Derrick"]
worker_id_2 = name_to_id["Ken"]
worker_id_3 = name_to_id["Sonny"]


def get_true_labels(min_length, answers, param1, param2):
    true_labels = []
    for i in range(min_length):
        annotation_data = answers.iloc[i]
        annotations = json.loads(annotation_data)
        is_outdoors = annotations[0][param1][param2]
        true_labels.append(1 if is_outdoors else 0)
    return true_labels


def plot_confusion_matrix_indoor_outdoor(
    y_true, y_pred, true_name, pred_name, save_path
):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Predicted {true_name} Indoors", f"Predicted {true_name} Outdoors"],
        columns=[f"Predicted {pred_name} Indoors", f"Predicted {pred_name} Outdoors"],
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("version 3 Indoor/Outdoor Confusion Matrix")
    plt.ylabel(f"{true_name}prediction")
    plt.xlabel(f"{pred_name} prediction")
    total_sum = np.sum(cm)
    plt.text(
        0.5,
        -0.15,
        f"Total: {total_sum}",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path)


def get_category_labels(min_length, answers, param1):
    category_labels = []
    for i in range(min_length):
        annotation_data = answers.iloc[i]
        annotations = json.loads(annotation_data)
        # Find the category that is marked as true
        for key, value in annotations[0][param1].items():
            if value:
                category_labels.append(int(key))
                break
    return category_labels


def plot_confusion_matrix_category(y_true, y_pred, true_name, pred_name, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    cm_df = pd.DataFrame(
        cm,
        index=[f" {category_mapping[i]}" for i in range(10)],
        columns=[f" {category_mapping[i]}" for i in range(10)],
    )
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Category Confusion Matrix")
    plt.ylabel(f"{true_name} Prediction")
    plt.xlabel(f"{pred_name} Prediction")
    total_sum = np.sum(cm)
    plt.text(
        0.5,
        -0.15,
        f"Total: {total_sum}",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def category(images_or_captions):
    param1 = "category"
    filename = f"{images_or_captions}3.csv"
    df = pd.read_csv(filename)

    answers_1 = df[df["WorkerId"] == worker_id_1]["Answer.taskAnswers"]
    answers_2 = df[df["WorkerId"] == worker_id_2]["Answer.taskAnswers"]

    save_path = f"{images_or_captions}_category_confusion_matrix.png"
    min_length = min(len(answers_1), len(answers_2))

    derrick_categories = get_category_labels(min_length, answers_1, param1)
    ken_categories = get_category_labels(min_length, answers_2, param1)

    plot_confusion_matrix_category(
        derrick_categories, ken_categories, "Derrick", "Ken", save_path
    )


def plot_confusion_matrix_manmade_natural(
    y_true, y_pred, true_name, pred_name, save_path
):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Predicted {true_name} Man-made", f"Predicted {true_name} Natural"],
        columns=[f"Predicted {pred_name} Man-made", f"Predicted {pred_name} Natural"],
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Version 3 Man-made/Natural Confusion Matrix")
    plt.ylabel(f"{true_name} Prediction")
    plt.xlabel(f"{pred_name} Prediction")
    total_sum = np.sum(cm)
    plt.text(
        0.5,
        -0.15,
        f"Total: {total_sum}",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path)


def manmade(images_or_captions):
    param1 = "type"
    param2 = "man-made"
    filename = f"{images_or_captions}3.csv"
    df = pd.read_csv(filename)

    answers_1 = df[df["WorkerId"] == worker_id_1]["Answer.taskAnswers"]
    answers_2 = df[df["WorkerId"] == worker_id_2]["Answer.taskAnswers"]
    answers_3 = df[df["WorkerId"] == worker_id_3]["Answer.taskAnswers"]
    # print(answers_1.iloc[0])
    save_path = f"{images_or_captions}_manmade_natural_confusion_matrix.png"
    min_length = min(len(answers_1), len(answers_2))
    derrick = get_true_labels(min_length, answers_1, param1, param2)
    ken = get_true_labels(min_length, answers_2, param1, param2)
    plot_confusion_matrix_manmade_natural(derrick, ken, "Derrick", "Ken", save_path)


def outdoors(images_or_captions):
    param1 = "location"
    param2 = "outdoors"

    filename = f"{images_or_captions}3.csv"
    df = pd.read_csv(filename)

    answers_1 = df[df["WorkerId"] == worker_id_1]["Answer.taskAnswers"]
    answers_2 = df[df["WorkerId"] == worker_id_2]["Answer.taskAnswers"]
    answers_3 = df[df["WorkerId"] == worker_id_3]["Answer.taskAnswers"]
    print(answers_1.iloc[0])
    save_path = f"{images_or_captions}_indoor_outdoor_confusion_matrix.png"
    min_length = min(len(answers_1), len(answers_2))

    derrick = get_true_labels(min_length, answers_1, param1, param2)
    ken = get_true_labels(min_length, answers_2, param1, param2)
    plot_confusion_matrix_indoor_outdoor(derrick, ken, "Derrick", "Ken", save_path)


def main():
    images_or_captions = "captions"

    manmade(images_or_captions)
    outdoors(images_or_captions)
    category(images_or_captions)


if __name__ == "__main__":
    main()
