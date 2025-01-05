
Project README

Overview

This project processes data to analyze verbs, generate counts, calculate KL divergence, and select captions based on hierarchical structures. Below is the order of execution for the scripts, along with a brief description of each step.

Workflow

1. Parse the Data
Run amr_parser.py to parse the input data and extract relevant structures.

python amr_parser.py
Purpose: Parses and preprocesses the input dataset for downstream analysis.
1.5 process verbs
1.75 analyze word counts and combinations. py
2. Pick Verbs
Run pick_verbs_and_generate_counts.py to extract verbs from the parsed data and generate initial counts.

python pick_verbs_and_generate_counts.py
Purpose: Identifies verbs in the dataset and calculates their frequencies.

3. Generate Counts and Analyze Distributions
Run count_topk_words_and_display_q1q2q3q4.py to generate word counts and analyze the distribution across quantiles.

python count_topk_words_and_display_q1q2q3q4.py
Purpose: Computes the frequency distribution of top-k words and visualizes their spread.

4. Calculate KL Divergence
Run calculate_kl_divergence.py to compute KL divergence values for the analyzed data.

python calculate_kl_divergence.py
Purpose: Measures the divergence between distributions to identify variability.

5. Split KL into Thirds
Run split_kl_into_thirds.py to divide the KL divergence values into three groups for hierarchical analysis.

python split_kl_into_thirds.py
Purpose: Categorizes KL values into thirds (low, medium, high) to facilitate downstream tasks.

6. Populate Hierarchy
Run populate_hierarchy.py to map the grouped data into a predefined hierarchy.

python populate_hierarchy.py
Purpose: Organizes data into a hierarchical structure based on KL divergence groupings.
7. Pick Captions
Run pick_captions.py to select captions containing location information based on the hierarchical analysis.

python pick_captions.py
Purpose: Filters captions by location relevance and hierarchical criteria.
Notes

Ensure all required dependencies are installed before running the scripts. You can install dependencies using:
pip install -r requirements.txt
The scripts should be executed in the order listed above for optimal results.
Modify parameters or input paths in each script as needed for your specific dataset.