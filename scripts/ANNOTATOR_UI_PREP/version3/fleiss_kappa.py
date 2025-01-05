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
    """
    Extract true labels based on a specific parameter.

    :param answers: Series with JSON annotation data.
    :param param1: First level key in the JSON structure.
    :param param2: Second level key under param1 in the JSON structure.
    :return: List of true labels.
    """
    true_labels = []
    for i in range(len(answers)):
        annotation_data = answers.iloc[i]
        try:
            annotations = json.loads(annotation_data)
            is_outdoors = annotations[0][param1][param2]
            true_labels.append(1 if is_outdoors else 0)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Error parsing annotation data for index {i}: {e}")
            true_labels.append(0)  # or another default value if needed
    return true_labels


def get_category_labels(answers, param1):
    """
    Extract category labels where the category is marked as true.

    :param answers: Series with JSON annotation data.
    :param param1: Key under which categories are listed in the JSON structure.
    :return: List of category labels.
    """
    category_labels = []
    for i in range(len(answers)):
        annotation_data = answers.iloc[i]
        try:
            annotations = json.loads(annotation_data)
            # Find the category that is marked as true
            for key, value in annotations[0][param1].items():
                if value:
                    category_labels.append(int(key))
                    break
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Error parsing annotation data for index {i}: {e}")
            category_labels.append(-1)  # Default or error-indicator value
    return category_labels
