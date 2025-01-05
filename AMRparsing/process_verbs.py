from tqdm import tqdm
import penman
from amr_parsing import find_n_edges, get_location_arguments, read_sentences_from_csv
import traceback
import pandas as pd
import os
import xml.etree.ElementTree as ET
import glob
from collections import defaultdict
import csv

def load_amr_graphs_from_directory(directory_path):
    amr_graphs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                amr_graphs.append(f.read())
    return amr_graphs

def process_verbs_and_save(sentences, categories, amr_graphs,
                           location_arguments, output_file):
    processed_data = []
    i = 1
    for sentence, category, amr_string in tqdm(
            zip(sentences, categories, amr_graphs),
            total=len(sentences),
            desc="Processing sentences"
    ):
        try:
            graph = penman.decode(amr_string)
            root_node = graph.top
            root_instances =[]
            for instance in graph.instances():
                if instance.source == root_node and instance.target[-2:].isnumeric():
                #if instance.target[-2:].isnumeric():
                    root_instances.append(instance)
           
          
            n_edges = find_n_edges(amr_string, location_arguments)
            for instance in root_instances:
                verb = instance.target
                source = instance.source
                lemma = verb.split("-")[0]  # Compute the lemma from the verb
                n_value = '+'.join([n[1][-1] for n in n_edges if n[0]==source])
                q1, q2, q3, q4 = category
                processed_data.append(
                    {
                        "ID": i,
                        "Sentence": sentence,
                        "Verb": verb,
                        "Lemma": lemma,
                        "PropBank Role (n)": n_value
                        or "N/A",  # Use "N/A" if no role number
                        "q1": q1,
                        "q2": q2,
                        "q3": q3,
                        "q4": q4
                    }
                )
        except Exception as e:
            print(f"Error processing AMR graph: {str(e)}")
            traceback.print_exc()
        i += 1
    df = pd.DataFrame(processed_data)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Processed verbs saved to {output_file}")

def main():
    """input file paths"""
    input_csv = "../data/sentences/sentences.csv"
    directory_for_original_graphs = "../data/amr_graphs_original25k"
    frames_folder = "../data/propbank-frames/frames"

    """output file paths"""
    output_verbs_csv = "../data/verbs/output_verbs.csv"

    original_amr_graphs = load_amr_graphs_from_directory(directory_for_original_graphs)
    location_arguments = get_location_arguments(frames_folder)
    sentences, categories = read_sentences_from_csv(input_csv)

    process_verbs_and_save(sentences, categories, original_amr_graphs,
                            location_arguments, output_verbs_csv)
    
if __name__ == "__main__":
    main()
