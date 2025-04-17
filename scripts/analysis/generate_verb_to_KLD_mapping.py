import csv
from Project1.scripts.helper.helper import strip_word

def load_kld_mapping(kld_csv_path, verbs):
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

def load_verbs(verb_csv_path):
    # Load only the 27 summary verbs
    verbs = set()
    with open(verb_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row['outer_key'].strip()
            if value!="outer_key":
                verbs.add(value)
    return verbs

def write_verb_kld_mapping(output_csv_path, verbs, kld_dict):
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['propbank_predicate', 'kld'])  # header
        for verb in sorted(verbs):  # optional: sorted output
            kld = kld_dict.get(verb, None)
            writer.writerow([verb, kld])

def main():
    ##input
    verb_csv_path = '../../data/results/original_captions_grouped_by_verb.csv'
    kld_csv_path = '../../data/kl_divergence/kl_divergence_by_row.csv'
    ##output
    output_csv_path = '../../data/kl_divergence/propbank_predicate_to_kld_mapping.csv'

    verbs = load_verbs(verb_csv_path)
    kld_dict = load_kld_mapping(kld_csv_path,verbs)
    write_verb_kld_mapping(output_csv_path, verbs, kld_dict)

if __name__ == '__main__':
    main()
