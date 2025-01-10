
Project README

Overview

This project processes data to analyze verbs, generate counts, calculate KL divergence, and select captions based on hierarchical structures. Below is the order of execution for the scripts, along with a brief description of each step.

Workflow

1. Parse the Data
Run amr_parser.py to parse the input data and extract relevant structures.

    ```python
    #input
    input_csv = "../data/sentences/sentences.csv"
    path_to_stog = "../models/model_stog"
    path_to_gtos = "../models/model_gtos"
    #output
    sentences_export25k.csv
    output_directory_for_original_graphs = "amr_graphs_original25k"
    output_directory_for_processed_graphs = "amr_graphs_processed25k"
    frames_folder = "/propbank-frames/frames"
    ```
    

Purpose: Parses and preprocesses the input dataset for downstream analysis.

2. process verbs: process_verbs.py from AMRparsing folder

    ```python
    #input
    input_csv = sentences.csv
    directory_for_original_graphs = "../../data/amr_graphs_original25k"
    frames_folder = "../../data/propbank-frames/frames"
    #output
    output_verbs_csv = "../../data/verbs/output_verbs.csv"
    ```

3. analyze word counts and combinations.py

    ```python
    #input
    word_counts_and_combinations_edited.csv"
    filtered_rows.csv"
    #output
    word_counts_and_combinations_edited.csv"
    ```

4. pick_verbs_and_generate_counts.py
Run pick_verbs_and_generate_counts.py to extract verbs from the parsed data and generate initial counts.

    ```python
    #input
    verb_path = "../../data/verbs/output_verbs.csv"
    #output
    target_verbs_csv = "../../data/verbs/target_verbs.csv"
    number_of_top_words = 300
    ```

python pick_verbs_and_generate_counts.py
Purpose: Identifies verbs in the dataset and calculates their frequencies.

5. count_topk_words_and_display_q1q2q3q4.py 
Run count_topk_words_and_display_q1q2q3q4.py to generate word counts and analyze the distribution across quantiles.

python count_topk_words_and_display_q1q2q3q4.py
Purpose: Computes the frequency distribution of top-k words and visualizes their spread.

    ```python
    #input
    file_path = "../../data/verbs/target_verbs.csv"
    all_combinations_path = "../../data/helper/combinations.yaml"
    #output
    verbs_path = "../../data/verbs/output_verbs.csv"
    output_csv_dir = "../../data/word_counts_and_combinations/"
    output_csv = "word_counts_and_combinations.csv"
    ```

6. Calculate KL Divergence
Run calculate_kl_divergence.py to compute KL divergence values for the analyzed data.

python calculate_kl_divergence.py
Purpose: Measures the divergence between distributions to identify variability.

    ```python
    #input
    #file_path = "../data/word_counts_and_combinations/word_counts_and_combinations_edited.csv"
    file_path = "../../data/word_counts_and_combinations/filtered_rows.csv"
    CFAC_path = "../../data/helper/counts_for_all_combinations.yaml"
    all_combinations_path = "../../data/helper/combinations.yaml"
    #output
    normalized_word_counts = "../../data/normalized_word_counts/normalized_word_counts.csv"
    kl_divergence_by_row = "../../data/kl_divergence/kl_divergence_by_row.csv"
    column_maxima_output = "../../data/column_maxima/column_maxima.csv"
    ```

7. Split KL into Thirds
Run split_kl_into_thirds.py to divide the KL divergence values into three groups for hierarchical analysis.

python split_kl_into_thirds.py
Purpose: Categorizes KL values into thirds (low, medium, high) to facilitate downstream tasks.

    ```python
    #input
    input_file = "../../data/kl_divergence/kl_divergence_by_row.csv"
    #output
    first = "../../data/kl_divergence/kl_divergence_by_row_top_third.csv"
    second = "../../data/kl_divergence/kl_divergence_by_row_second_third.csv"
    third = "../../data/kl_divergence/kl_divergence_by_row_third_third.csv"
    ```

8. Populate Hierarchy

Run populate_hierarchy.py to map the grouped data into a predefined hierarchy.

python populate_hierarchy.py
Purpose: Organizes data into a hierarchical structure based on KL divergence groupings.

    ```python
    KLD = "../../data/kl_divergence/"
    input
    input_path = f"{KLD}kl_divergence_by_row"
    #output
    output_path = f"{KLD}hierarchy_populated"
    ```

9. Pick Captions

Run pick_captions.py to select captions containing location information based on the hierarchical analysis.

python pick_captions.py
Purpose: Filters captions by location relevance and hierarchical criteria.

    ```python
    #input
    csv_files = [
        '../../data/kl_divergence/hierarchy_populated_top_third.csv',
        '../../data/kl_divergence/hierarchy_populated_second_third.csv',
        '../../data/kl_divergence/hierarchy_populated_third_third.csv'
    ]
    exported_sentences_path = "../../data/exported_sentences/sentences_export25k.csv"
    output_verbs_path = "../../data/verbs/output_verbs.csv"
    #output```python
    combined_data_path = "../../data/combined_data/combined_data.csv"
    picked_captions_path = "../../data/picked_captions/picked_captions.csv"
    ```

NOTES

Ensure all required dependencies are installed before running the scripts. You can install dependencies using:
pip install -r requirements.txt
The scripts should be executed in the order listed above for optimal results.
Modify parameters or input paths in each script as needed for your specific dataset.