import json
import yaml
import pandas as pd


def load_config(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def pretty_print_matrix(matrix):
    df = pd.DataFrame(matrix)
    print(df.to_string(index=False, header=False))


def read_answers(df, worker_id):
    return df[df["WorkerId"] == worker_id]["Answer.taskAnswers"]


def get_true_labels(answers, param1, param2):
    true_labels = []
    for i in range(len(answers)):
        annotation_data = answers.iloc[i]
        annotations = json.loads(annotation_data)
        is_outdoors = annotations[0][param1][param2]
        true_labels.append(1 if is_outdoors else 0)
    return true_labels


def get_category_labels(answers, param1):
    category_labels = []
    for i in range(len(answers)):
        annotation_data = answers.iloc[i]
        annotations = json.loads(annotation_data)
        # Find the category that is marked as true
        for key, value in annotations[0][param1].items():
            if value:
                category_labels.append(int(key))
                break
    return category_labels
