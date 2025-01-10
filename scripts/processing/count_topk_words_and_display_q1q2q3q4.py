import pandas as pd
from collections import defaultdict
import csv
import yaml


def generate_word_counts(output_verbs, sentences,img_path):
    caption_counts = defaultdict(lambda: defaultdict(int))
    for index, row in output_verbs.iterrows():
        categories = (row['q1'], row['q2'], str(row['q3']), str(row['q4']))
        combination = ' - '.join(categories).replace(' - nan', '')
        caption_counts[combination][row['Verb']] += 1

    merged = output_verbs.merge(sentences, on='ID')
    processed = merged.loc[merged['Processed Sentence'] != 'skip_because_no_change']

    image_sets = defaultdict(lambda: defaultdict(set))
    for index, row in processed.iterrows():
        categories = (row['q1'], row['q2'], str(row['q3']), str(row['q4']))
        combination = ' - '.join(categories).replace(' - nan', '')
        image_sets[combination][row['Verb']].add(row['Image ID'])

    image_counts = defaultdict(lambda: defaultdict(int))
    for combination in image_sets:
        for verb in image_sets[combination]:
            image_counts[combination][verb] = len(image_sets[combination][verb])

    image_counts_and_combinations = pd.DataFrame.from_dict(image_counts)
    image_counts_and_combinations.to_csv(img_path)

    word_counts_and_combinations = pd.DataFrame.from_dict(caption_counts)
    target_verbs = image_counts_and_combinations.loc[
        image_counts_and_combinations.max(axis=1) >= 5].index
    word_counts_and_combinations = word_counts_and_combinations.loc[target_verbs]
    return word_counts_and_combinations
   

    
def main():
    """input and output file paths"""
    #input
    output_verbs_path = "../../data/verbs/output_verbs.csv"
    sentences_export_path = "../../data/exported_sentences/sentences_export25k.csv"
    img_path = "../../data/image_counts_and_combinations/image_counts_and_combinations.csv"
    #output
    output_csv= "../../data/word_counts_and_combinations/word_counts_and_combinations.csv"
    """input and output file paths"""

    output_verbs = pd.read_csv(output_verbs_path)
    sentences = pd.read_csv(sentences_export_path)
    sentences['ID'] = range(1, 25341)
    sentences['Image ID'] = (sentences['ID'] - 1) // 5

    word_counts_and_combinations = generate_word_counts(output_verbs, sentences,img_path)
    word_counts_and_combinations.rename_axis('Word', inplace=True)
    word_counts_and_combinations.to_csv(output_csv)

if __name__ == "__main__":
    main()
