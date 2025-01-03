import spacy
import amrlib
import os
import csv
from tqdm import tqdm
import penman
import pandas as pd
import glob
import xml.etree.ElementTree as ET
from collections import defaultdict
import os
print("Current Working Directory:", os.getcwd())
os.environ["HF_HOME"] = "/home/egk265/.cache/huggingface"


spacy.load("en_core_web_sm")

def find_n_edges(amr_string, location_arguments):
    # Decode the AMR graph
    graph = penman.decode(amr_string)
    # Extract verbs from the AMR graph
    n_edges = []
    for instance in graph.instances():
        if instance.target[-2:].isnumeric():
            verb = instance.target
            source = instance.source
            # Iterate over verbs and check against PropBank frames
            if verb in location_arguments:
                for n in location_arguments[verb]:
                    for edge in graph.edges():
                        if edge[0] == source and edge[1] == f":ARG{n}":
                            n_edges.append(edge)
    return n_edges


def get_location_arguments(frames_folder):
    location_arguments = defaultdict(set)
    # Use glob to find all XML files in the directory
    xml_files = glob.glob(f"{frames_folder}/*.xml")
    for xml_file in xml_files:
        try:
            # Parse the XML file
            tree = ET.parse(xml_file)
            root = tree.getroot()
            rolesets = root.findall(".//roleset")
            for roleset in rolesets:
                roleset_id = roleset.get("id").replace('.','-').replace('_','-')
                # Search for elements with f="LOC" attribute
                role_elements = roleset.findall(".//role")
                for role in role_elements:
                    # Check if the 'f' attribute is 'LOC'
                    if role.get("f") == "LOC":
                        location_arguments[roleset_id].add(role.get("n"))
        except ET.ParseError:
            print(f"Error parsing file: {xml_file}")
    return location_arguments


def export_sentences_to_csv(original_sents, processed_sents, output_file):
    data = {
        "Original Sentence": [
            sent if sent != "skip" else "skip" for sent in original_sents
        ],
        "Processed Sentence": [
            sent if sent != "skip" else "skip" for sent in processed_sents
        ],
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding="utf-8")


def read_sentences_from_csv(file_path):
    sentences = []
    categories = []
    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header if present
        for row in reader:
            if row:
                # Assuming sentences are in the first column
                sentences.append(row[0])
                categories.append(row[1:5])
    return sentences, categories


def stog(sentences,path_to_stog):
    stog = amrlib.load_stog_model(path_to_stog)
    graphs = []
    # Initialize a progress bar with tqdm
    for sentence in tqdm(sentences, desc="Generating graphs", unit="sentence"):
        # Generate AMR graph for one sentence
        graph = stog.parse_sents([sentence])[0]
        graphs.append(graph)
    return graphs


def sentence_to_graph(sentences, path_to_stog):
    amr_graphs = stog(sentences, path_to_stog)
    print(f"Generated {len(amr_graphs)} AMR graphs")
    return amr_graphs


def find_type_edges(amr_string, type):
    location_edges = []
    try:
        graph = penman.decode(amr_string)
        for triple in graph.triples:
            source, role, target = triple
            if role in type:
                location_edges.append((source, target))
    except Exception as e:
        print(f"Error decoding AMR string: {e}")
        print("AMR string:", amr_string)
    return location_edges


def filter_branches(branches, location_nodes):
    new_branches = []
    for role, target in branches:
        if target[0] not in location_nodes:
            if isinstance(target, tuple):
                # Recursively filter sub-branches
                new_target = (target[0], filter_branches(target[1], location_nodes))
                new_branches.append((role, new_target))
            else:
                new_branches.append((role, target))
    return new_branches

def remove_location_or_argument(amr_string, location_arguments):
    graph = penman.decode(amr_string)
    top = graph.top
    tree = penman.parse(amr_string)
    prefix_to_concept = {}
    # Populate the prefix-to-concept mapping
    for path, branch in tree.walk():
        role, target = branch
        if role == '/': #concept definition
            prefix_to_concept[tuple(path)] = target
    
    # Step 1: Identify location-related nodes
    location_paths = []
    roles_to_remove = [":location"]
    for path, branch in tree.walk():
        role,target = branch
        # Check if the edge role is location-related or its target is a location-related node
        #if role in [":location", ":location-of"] or role in location_arguments:
        if role in roles_to_remove:
            location_paths.insert(0, path)
        temp = path[:-1] + (0,)
        parent_prefix = tuple(temp)
        parent_concept = prefix_to_concept.get(parent_prefix, None)
        # Check if the role is in location_arguments for the parent concept
        if parent_concept and parent_concept in location_arguments:
            role_number = role.replace(':ARG', '')
            if role_number in location_arguments[parent_concept]:
                location_paths.insert(0,path)
    top = tree.nodes()[0]
    for path in location_paths:
        node = top
        for index in path[:-1]:
            node = node[1][index][1]
        node[1].pop(path[-1])

    new_tree = penman.Tree(top)
    new_amr = penman.format(new_tree)
    return new_amr



def process_graphs(amr_graphs, location_arguments,path_to_gtos):
    processed_graphs = []
    gtos_model = amrlib.load_gtos_model(path_to_gtos)
    processed_sentences = []
    for i, graph in enumerate(amr_graphs):
        location_edges = find_type_edges(graph,[":location", ":location-of"])
        n = find_n_edges(graph, location_arguments)
        if location_edges or n:
            print("original graph")
            print(graph)
            processed_graph = remove_location_or_argument(
                graph, location_arguments
                )
            print("processed graph")
            print(processed_graph)
            processed_sentence=gtos_model.generate([processed_graph])[0][0]
            processed_sentences.append(processed_sentence)
        else:
            processed_sentences.append("skip_because_no_change")
            processed_graph = None
        processed_graphs.append(processed_graph)
    return processed_graphs, processed_sentences


def save_graphs_to_directory(graphs, output_dir, prefix="processed"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, graph in enumerate(graphs):
        if graph is None:  # Skip None entries
            print(f"Skipping graph {i+1} as it is None.")
            continue
        file_path = os.path.join(output_dir, f"{prefix}_amr_graph_{i+1}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(graph)
        print(f"Saved {prefix} graph {i+1} to {file_path}")




def main():
    """input file paths"""
    input_csv = "../data/sentences/sentences.csv"
    path_to_stog = "../models/model_stog"
    path_to_gtos = "../models/model_gtos"

    """output file paths"""
    output_verbs_csv = "../data/verbs/output_verbs.csv"
    output_file_exported_sentences = "../data/exported_sentences/sentences_export25k.csv"
    output_directory_for_original_graphs = "../data/amr_graphs_original25k"
    output_directory_for_processed_graphs = "../data/amr_graphs_processed25k"
    frames_folder = "../data/propbank-frames/frames"


    location_arguments = get_location_arguments(frames_folder)
    sentences, categories = read_sentences_from_csv(input_csv)
    print(f"Read {len(sentences)} sentences from {input_csv}")
    amr_graphs = sentence_to_graph(sentences,path_to_stog)
    save_graphs_to_directory(amr_graphs, output_directory_for_original_graphs, prefix="original")
    processed_graphs, processed_sents = process_graphs(amr_graphs, location_arguments,path_to_gtos)
    save_graphs_to_directory(processed_graphs, output_directory_for_processed_graphs, prefix = "processed" )
    export_sentences_to_csv(sentences, processed_sents, output_file_exported_sentences)
    


if __name__ == "__main__":
    main()
