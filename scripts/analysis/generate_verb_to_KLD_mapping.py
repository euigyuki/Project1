import csv
from dataclasses import dataclass
from pathlib import Path



def load_kld_mapping(path_config, verbs):
    kld_csv_path = path_config.kld_csv_path
    kld_dict = {}
    with open(kld_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            propbank_predicate = row['Word'].strip()
            if propbank_predicate in verbs:
                try:
                    kld = float(row['KLD_SUM'])
                    kld_dict[propbank_predicate] = kld
                except ValueError:
                    continue
    return kld_dict

def load_verbs(path_config):
    verb_csv_path = path_config.verb_csv_path
    # Load only the 27 summary verbs
    verbs = set()
    with open(verb_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row['outer_key'].strip()
            if value!="outer_key":
                verbs.add(value)
    return verbs

def write_verb_kld_mapping(path_config, verbs, kld_dict):
    output_csv_path = path_config.output_csv_path
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['propbank_predicate', 'kld'])  # header
        for verb in sorted(verbs):  # optional: sorted output
            kld = kld_dict.get(verb, None)
            writer.writerow([verb, kld])


@dataclass
class PathConfig:
    verb_csv_path: Path
    kld_csv_path: Path
    output_csv_path: Path
    
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"    

def main():
    path_config = PathConfig(
        ## Input paths
        verb_csv_path=DATA_DIR / "results" / "original_captions_grouped_by_verb.csv",
        kld_csv_path=DATA_DIR / "kl_divergence" / "kl_divergence_by_row.csv",
        ## Output paths
        output_csv_path=DATA_DIR / "kl_divergence" / "propbank_predicate_to_kld_mapping.csv"
    )
    
    verbs = load_verbs(path_config)
    kld_dict = load_kld_mapping(path_config, verbs)
    write_verb_kld_mapping(path_config, verbs, kld_dict)

if __name__ == '__main__':
    main()
