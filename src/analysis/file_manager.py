import json
import csv
import pandas as pd

class FileManager:
    @staticmethod
    def output_to_csv(data, output_csv):
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Caption', 'KLD Human to LLM', 'KLD LLM to Human',
                             'Jensen-Shannon Divergence', 'Human Probs', 'LLM Probs'])
            for caption, divergence in data.items():
                writer.writerow([
                    caption,
                    divergence["kl_div_human_llm"],
                    divergence["kl_div_llm_human"],
                    divergence['js_div'],  # only write the numeric value here
                    json.dumps(divergence['human_probs']),
                    json.dumps(divergence['llm_probs'])
                ])
        print(f"Data has been written to {output_csv}")

    @staticmethod
    def calculate_average_divergence(csv_file):
        df = pd.read_csv(csv_file)
        avg = df['Jensen-Shannon Divergence'].mean()
        print(f"Average Jensen-Shannon Divergence in {csv_file}: {avg:.4f}")
        return avg