import csv
import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SANITY_DIR = PROJECT_ROOT / "data" / "results" / "sanity_checks"
JUDGEMENTS_DIR = PROJECT_ROOT / "data" / "results" / "judgements_for_ken2"  # adjust if different

def filter_probs_sum4(filename):
    count = 0
    total = 0
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            total += 1
            prob_list = ast.literal_eval(row[2])
            if sum(prob_list) == 4:
                count += 1
            # else:
            #     print(f"Row with sum != 4: {row}")
    print(f"Total matches found: {count}, Total rows: {total}")

def run_group(title: str, base_dir: Path, files: list[str]) -> None:
    print(f"\n=== {title} ===")
    for name in files:
        path = base_dir / name
        if not path.exists():
            print(f"{path} (missing)")
            continue
        print(f"\n--- {path.name} ---")
        filter_probs_sum4(path)

if __name__ == "__main__":
    SANITY_FILES = [
        "midjourney_sanity_check.csv",
        "flux_sanity_check.csv",
        "dalle_sanity_check.csv",
    ]
    JUDGEMENT_FILES = [
        "dalle_judgements_for_ken2.csv",
        "midjourney_judgements_for_ken2.csv",
        "flux_judgements_for_ken2.csv",
    ]

    run_group("Sanity checks", SANITY_DIR, SANITY_FILES)
    run_group("Judgements for Ken2", JUDGEMENTS_DIR, JUDGEMENT_FILES)
