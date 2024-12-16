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
from collections import defaultdict

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


def stog(sentences):
    stog = amrlib.load_stog_model()
    graphs = []
    # Initialize a progress bar with tqdm
    for sentence in tqdm(sentences, desc="Generating graphs", unit="sentence"):
        # Generate AMR graph for one sentence
        graph = stog.parse_sents([sentence])[0]
        graphs.append(graph)
    return graphs


def sentence_to_graph(sentences):
    amr_graphs = stog(sentences)
    print(f"Generated {len(amr_graphs)} AMR graphs")
    return amr_graphs


def find_type_edges(amr_string, type):
    location_edges = []
    try:
        graph = penman.decode(amr_string)
        for triple in graph.triples:
            source, role, target = triple
            if role == type:
                location_edges.append((source, target))
    except Exception as e:
        print(f"Error decoding AMR string: {e}")
        print("AMR string:", amr_string)
    return location_edges


def remove_location_or_argument(amr_string, location_arguments):
    graph = penman.decode(amr_string)
    instances = graph.instances()
    edges = graph.edges()
    attributes = graph.attributes()
    top = graph.top
    n = find_n_edges(amr_string, location_arguments)
    # Step 1: Remove location-related edges
    non_location_edges = []
    for edge in edges:
        if edge[1] != ":location" and edge not in n:
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
    # Step 4: Keep only instances and attributes with reachable nodes
    new_instances = []
    for instance in instances:
        if instance.source in reachable:
            new_instances.append(instance)
    new_attributes = []
    for attribute in attributes:
        if attribute.source in reachable:
            new_attributes.append(attribute)
    new_graph = new_instances + new_triples + new_attributes
    new_graph = penman.Graph(new_graph, top=top)
    try:
        return penman.encode(new_graph)
    except penman.exceptions.LayoutError:
        print("Warning: Could not encode modified graph. Returning original.")
        return amr_string


def process_graphs(amr_graphs, location_arguments):
    processed_graphs = []
    gtos_model = amrlib.load_gtos_model()
    processed_sentences = []
    for i, graph in enumerate(amr_graphs):
        try:
            location_edges = find_type_edges(graph, ":location")
            n = find_n_edges(graph, location_arguments)
            if location_edges or n:
                processed_graph = remove_location_or_argument(
                    graph, location_arguments
                    )
                processed_sentence=gtos_model.generate([processed_graph])[0][0]
                processed_sentences.append(processed_sentence)
            else:
                processed_sentences.append("skip_because_no_change")
                processed_graph = None
            processed_graphs.append(processed_graph)
        except Exception as e:
            print(f"Error processing graph {i+1}: {str(e)}")
            print(f"Exception type: {type(e)}")
            traceback.print_exc()
            processed_sentences.append('skip_because_error')
            processed_graphs.append(
                graph
            )  # Keep the original graph if there's an error
    return processed_graphs, processed_sentences


def save_graphs_to_directory(graphs, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, graph in enumerate(graphs):
        if graph is None:  # Skip None entries
            print(f"Skipping graph {i+1} as it is None.")
            continue
        file_path = os.path.join(output_dir, f"amr_graph_{i+1}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(graph)
        print(f"Saved processed graph {i+1} to {file_path}")


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
            instances = [
                instance
                for instance in graph.instances()
                if instance.target[-2:].isnumeric()
            ]
            n_edges = find_n_edges(amr_string, location_arguments)
            for instance in instances:
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
    """input and output file paths"""
    input_csv = "../data/sentences/sentences.csv"
    output_verbs_csv = "../data/verbs/output_verbs.csv"
    output_file = "sentences_export.csv"
    output_directory_for_graphs = "../data/amr_graphs"
    """input and output file paths"""
    # Specify the path to your frames folder
    frames_folder = "../data/propbank-frames/frames"
    location_arguments = get_location_arguments(frames_folder)

    sentences, categories = read_sentences_from_csv(input_csv)
    print(f"Read {len(sentences)} sentences from {input_csv}")
    amr_graphs = sentence_to_graph(sentences)
    graphs, processed_sents = process_graphs(amr_graphs, location_arguments)
    save_graphs_to_directory(graphs, output_directory_for_graphs)
    export_sentences_to_csv(sentences, processed_sents, output_file)
    process_verbs_and_save(sentences, categories, amr_graphs,
                           location_arguments, output_verbs_csv)


if __name__ == "__main__":
    main()
