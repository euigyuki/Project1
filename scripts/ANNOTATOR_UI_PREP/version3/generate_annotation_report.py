import pandas as pd
import json
from helper import load_config

config = load_config("config.yaml")

category_mapping = config["category_mapping"]
id_mapping = config["id_mapping"]

# Read the CSV file
df = pd.read_csv("captions3.csv")


# Function to parse JSON string
def parse_json(json_str):
    try:
        return json.loads(json_str)
    except:
        return {}


def get_category(category_dict):
    for k, v in category_dict.items():
        if v:
            return config["category_mapping"].get(k, f"Unknown Category {k}")
    return "Unknown"


# Process the data
data = []
for _, row in df.iterrows():
    answers = parse_json(row["Answer.taskAnswers"])
    if answers and isinstance(answers, list) and len(answers) > 0:
        answer = answers[0]
        category = next(
            (
                category_mapping[int(k)]
                for k, v in answer.get("category", {}).items()
                if v
            ),
            "Unknown",
        )
        entry = {
            "Caption": row["Input.sentence"],
            "Annotator": config["id_mapping"][row["WorkerId"]],
            "Category": category,
            "Location": "Outdoors"
            if answer.get("location", {}).get("outdoors")
            else "Indoors",
            "Type": "Man-made" if answer.get("type", {}).get("man-made") else "Natural",
            "Reasoning": answer.get("reasoning", ""),
        }
        data.append(entry)


# Create a DataFrame
result_df = pd.DataFrame(data)

# Display the result
print(result_df)

# Save to CSV
result_df.to_csv("annotations_report.csv", index=False)
