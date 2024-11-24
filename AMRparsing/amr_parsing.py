import spacy
import amrlib
import os
import csv
from tqdm import tqdm
import penman
import traceback
import pandas as pd
import glob
import xml.etree.ElementTree as ET

spacy.load("en_core_web_sm")


def return_n_of_amr_string(amr_string):
    # Decode the AMR graph
    graph = penman.decode(amr_string)
    # Extract verbs from the AMR graph
    verbs = []
    for instance in graph.instances():
        if "-01" in instance.target:
            verb = instance.target
            verbs.append(verb)
    # Iterate over verbs and check against PropBank frames
    for verb in verbs:
        lemma = verb.split("-")[
            0
        ]  # Extract lemma from verb (e.g., "walk" from "walk-01")
        # Specify the path to your frames folder
        frames_folder = "../data/propbank-frames/frames"
        # Use glob to find all XML files in the directory
        xml_files = glob.glob(f"{frames_folder}/*.xml")
        for xml_file in xml_files:
            filename = os.path.splitext(os.path.basename(xml_file))[0]
            try:
                # Parse the XML file
                tree = ET.parse(xml_file)
                root = tree.getroot()
                # Search for elements with f="LOC" attribute
                role_elements = root.findall(".//role")
                for role in role_elements:
                    # Check if the 'f' attribute is 'LOC'
                    if role.get("f") == "LOC" and lemma == filename:
                        n_value = role.get("n")
                        return n_value
            except ET.ParseError:
                print(f"Error parsing file: {xml_file}")
    return None


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
    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header if present
        for row in reader:
            if row:
                # Assuming sentences are in the first column
                sentences.append(row[0])
    return sentences


def stog(sentences):
    stog = amrlib.load_stog_model(
        model_dir="amrlib/data/model_stog"
    )
    graphs = []
    # Initialize a progress bar with tqdm
    for sentence in tqdm(sentences, desc="Generating graphs", unit="sentence"):
        # Generate AMR graph for one sentence
        graph = stog.parse_sents([sentence])[0]
        graphs.append(graph)
    return graphs


def gtos(graph_strings, gtos):
    gtos = amrlib.load_gtos_model(
        model_dir="amrlib/data/model_gtos"
    )
    sentences = []
    print("\nGraphs to Sentences:")
    for graph_string in graph_strings:
        location_nodes = find_type_nodes(graph_string, ":location")
        n_argument = return_n_of_amr_string(graph_string)
        if location_nodes or n_argument is not None:
            original_sentence = gtos.generate([graph_string])[0]
            print(original_sentence)
            modified_graph_string = remove_location_or_argument(graph_string)
            processed_sentence = gtos.generate([modified_graph_string])[0]
            print(processed_sentence)
            sentences.append(processed_sentence)
        else:
            original_sentence = gtos.generate([graph_string])[0]
            sentences.append(original_sentence)
    return sentences


def sentence_to_graph(input_csv):
    sentences = read_sentences_from_csv(input_csv)
    print(f"Read {len(sentences)} sentences from {input_csv}")
    amr_graphs = stog(sentences)
    print(f"Generated {len(amr_graphs)} AMR graphs")
    return amr_graphs


def find_type_nodes(amr_string, type):
    location_nodes = []
    try:
        graph = penman.decode(amr_string)
        for triple in graph.triples:
            source, role, target = triple
            if role == type:
                location_nodes.append((source, target))
    except Exception as e:
        print(f"Error decoding AMR string: {e}")
        print("AMR string:", amr_string)
    return location_nodes


def remove_location_or_argument(amr_string):
    graph = penman.decode(amr_string)
    instances = graph.instances()
    edges = graph.edges()
    attributes = graph.attributes()
    top = graph.top
    n = return_n_of_amr_string(amr_string)
    print("argument number ", n)
    # Step 1: Remove location-related edges
    non_location_edges = []
    for edge in edges:
        if edge[1] != ":location" and edge[1] != f":ARG{n}":
            non_location_edges.append(edge)
    # Step 2: Find reachable nodes using BFS
    reachable = set([top])
    queue = [top]
    while queue:
        node = queue.pop(0)
        for s, _, t in non_location_edges:
            if s == node and t not in reachable:
                reachable.add(t)
                queue.append(t)
            elif t == node and s not in reachable:
                reachable.add(s)
                queue.append(s)
            elif s in reachable and t not in reachable:
                reachable.add(t)
                queue.append(t)
            elif t in reachable and s not in reachable:
                reachable.add(s)
                queue.append(s)
    # Step 3: Keep only triples with reachable nodes
    new_triples = []
    for t in non_location_edges:
        if t[0] in reachable and (isinstance(t[2], str) or t[2] in reachable):
            new_triples.append(t)
    # Step 4: Keep only instances with reachable nodes
    new_instances = []
    for instance in instances:
        if instance.source in reachable:
            new_instances.append(instance)
    new_graph = new_instances + new_triples + attributes
    new_graph = penman.Graph(new_graph, top=top)
    try:
        return penman.encode(new_graph)
    except penman.exceptions.LayoutError:
        print("Warning: Could not encode modified graph. Returning original.")
        return amr_string


def process_and_save_graphs(amr_graphs, output_dir, output_file_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    processed_graphs = []
    gtos_model = amrlib.load_gtos_model(
        "amrlib/data/model_gtos"
    )
    original_sentences = []
    processed_sentences = []
    for i, graph in enumerate(amr_graphs):
        try:
            location_nodes = find_type_nodes(graph, ":location")
            n = return_n_of_amr_string(graph)
            if location_nodes or n is not None:
                original_sentence = gtos_model.generate([graph])[0]
                original_sentences.append(original_sentence)
                processed_graph = remove_location_or_argument(graph)
                processed_sentence = gtos_model.generate([processed_graph])[0]
                processed_sentences.append(processed_sentence)
            else:
                original_sentence = gtos_model.generate([graph])[0]
                original_sentences.append("skip_because_no_change")
                processed_sentences.append("skip")
                processed_graph = None
            processed_graphs.append(processed_graph)
        except Exception as e:
            print(f"Error processing graph {i+1}: {str(e)}")
            print(f"Exception type: {type(e)}")
            traceback.print_exc()
            processed_graphs.append(
                graph
            )  # Keep the original graph if there's an error
    return processed_graphs, original_sentences, processed_sentences


def save_graphs_to_directory(graphs, output_dir):
    for i, graph in enumerate(graphs):
        if graph is None:  # Skip None entries
            print(f"Skipping graph {i+1} as it is None.")
            continue
        file_path = os.path.join(output_dir, f"amr_graph_{i+1}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(graph)
        print(f"Saved processed graph {i+1} to {file_path}")


def process_verbs_and_save(sentences, amr_graphs, output_file):
    processed_data = []
    for sentence, amr_string in zip(sentences, amr_graphs):
        try:
            graph = penman.decode(amr_string)
            verbs = [
                instance.target
                for instance in graph.instances()
                if "-01" in instance.target
            ]
            for verb in verbs:
                lemma = verb.split("-")[0]  # Compute the lemma from the verb
                n_value = return_n_of_amr_string(amr_string)
                processed_data.append(
                    {
                        "Sentence": sentence,
                        "Verb": verb,
                        "Lemma": lemma,
                        "PropBank Role (n)": n_value
                        or "N/A",  # Use "N/A" if no role number
                    }
                )
        except Exception as e:
            print(f"Error processing AMR graph: {str(e)}")
            traceback.print_exc()
    df = pd.DataFrame(processed_data)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Processed verbs saved to {output_file}")


def main():
    """input and output file paths"""
    input_csv = "../data/sentences/sentences_edited_first50.csv"
    output_verbs_csv = "../data/verbs/output_verbs.csv"
    output_file = "sentences_export_first50.csv"
    output_directory_for_graphs = "../data/amr_graphs"
    """input and output file paths"""

    amr_graphs = sentence_to_graph(input_csv)
    graphs, original_sents, processed_sents = process_and_save_graphs(
        amr_graphs, output_directory_for_graphs, output_file
    )
    save_graphs_to_directory(graphs, output_directory_for_graphs)
    export_sentences_to_csv(original_sents, processed_sents, output_file)
    sentences = read_sentences_from_csv(input_csv)
    process_verbs_and_save(sentences, amr_graphs, output_verbs_csv)


if __name__ == "__main__":
    main()
