import json
import pandas as pd
from dataclasses import dataclass
from src.helper.helper_functions import AnnotationProcessor
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from data.results.captions_annotated_by_humans.organize_captions import extract_annotations
from typing import List

def extract_llm_annotations(path_config) -> pd.DataFrame:
    model_to_path = {
        "chatgpt": path_config.chatgpt,
        "claude": path_config.claude,
        "perplexity": path_config.perplexity,
        "deepseek": path_config.deepseek,
    }

    rows = []
    for model_name, filepath in model_to_path.items():
        with open(filepath, "r") as f:
            data = json.load(f)
            for entry in data:
                sentence = entry.get("caption", "")
                answer = entry.get("annotation", {})
                rows.append({
                    "WorkerId": model_name,
                    "Input.sentence": sentence,
                    "Answer.taskAnswers": AnnotationProcessor.process_llm_annotation(answer)
                })
    df = pd.DataFrame(rows)
    return df


@dataclass
class PathConfig:
    input_csvs: List[Path]
    chatgpt: Path
    claude: Path
    perplexity: Path
    deepseek: Path
    output_csv: Path  


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT


def main():
    path_config = PathConfig(
        input_csvs= None,
        chatgpt=DATA_DIR/"results"/"llm_annotations"/"chatgpt_annotations.json",
        claude=DATA_DIR/"results"/"llm_annotations"/"claude_annotations.json",
        perplexity=DATA_DIR/"results"/"llm_annotations"/"perplexity_annotations.json",
        deepseek=DATA_DIR/"results"/"llm_annotations"/"deepseek_annotations.json",
        output_csv=DATA_DIR/"results"/"llm_annotations"/"captions_annotated_by_llms.csv"
    )   
  

    extract_annotations(path_config,extract_llm_annotations, "Input.sentence", entity="llms")


if __name__ == "__main__":
    main()
