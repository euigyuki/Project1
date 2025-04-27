import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import List


class VerbAccuracyPlotter:
    def __init__(self, path_config):
        self.path_config = path_config
        self.df_human = pd.read_csv(path_config.x_over_20_for_human_annotators)
        self.df_vlm = pd.read_csv(path_config.x_over_20_for_human_vlms)
        self.df_llm = pd.read_csv(path_config.x_over_20_for_human_llms)
        self.verbs = sorted(set(self.df_human["Verb"].dropna()))
        self.groups = self._group_verbs(self.verbs)

    def _group_verbs(self, verbs: List[str], group_size: int = 9) -> List[List[str]]:
        return [verbs[i:i + group_size] for i in range(0, len(verbs), group_size)]

    def plot_all(self):
        for idx, group in enumerate(self.groups):
            for source in ["Human", "LLM", "VLM"]:
                self._plot_source_comparison(group, idx, source)
                
    def _plot_source_comparison(self, verb_group: List[str], group_index: int, source: str):
        if source == "Human":
            df = self.df_human
        elif source == "LLM":
            df = self.df_llm
        elif source == "VLM":
            df = self.df_vlm
        else:
            raise ValueError(f"Invalid source: {source}")

        data = {
            "Verb": verb_group,
            "Original": [df[df["Verb"] == v]["Original percentage"].values[0] for v in verb_group],
            "Processed": [df[df["Verb"] == v]["Processed percentage"].values[0] for v in verb_group],
        }
        df_plot = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.3
        x = range(len(df_plot))

        ax.bar([i - bar_width / 2 for i in x], df_plot["Original"], width=bar_width, label="Original")
        ax.bar([i + bar_width / 2 for i in x], df_plot["Processed"], width=bar_width, label="Processed")

        ax.set_xticks(x)
        ax.set_xticklabels(df_plot["Verb"], rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{source}: Original vs. Processed – Group {group_index + 1}")
        ax.legend()

        plt.tight_layout()
        output_file = self.path_config.output_path / f"group_{group_index + 1}_{source.lower()}_original_vs_processed.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_file}")


    def _plot_group(self, verb_group: List[str], group_index: int, column: str):
        data = {
            "Verb": verb_group,
            "Human": [self.df_human[self.df_human["Verb"] == v][column].values[0] for v in verb_group],
            "LLM": [self.df_llm[self.df_llm["Verb"] == v][column].values[0] for v in verb_group],
            "VLM": [self.df_vlm[self.df_vlm["Verb"] == v][column].values[0] for v in verb_group],
        }
        df_plot = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.2
        x = range(len(df_plot))

        ax.bar([i - bar_width for i in x], df_plot["Human"], width=bar_width, label="Human")
        ax.bar(x, df_plot["LLM"], width=bar_width, label="LLM")
        ax.bar([i + bar_width for i in x], df_plot["VLM"], width=bar_width, label="VLM")

        ax.set_xticks(x)
        ax.set_xticklabels(df_plot["Verb"], rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{column} – Group {group_index + 1}")
        ax.legend()

        plt.tight_layout()
        output_file = self.path_config.output_path / f"group_{group_index + 1}_{column.replace(' ', '_').lower()}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_file}")

@dataclass
class PathConfig:
    x_over_20_for_human_annotators: Path
    x_over_20_for_human_vlms: Path
    x_over_20_for_human_llms: Path
    output_path: Path

# Paths setup
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"/"results"/"x_over_20"

def main():
    path_config = PathConfig(
        x_over_20_for_human_annotators=DATA_DIR / "x_over_20_for_human_annotators.csv",
        x_over_20_for_human_vlms=DATA_DIR / "x_over_20_for_vlms.csv",
        x_over_20_for_human_llms=DATA_DIR / "x_over_20_for_llms.csv",
        output_path = DATA_DIR 
    )
    plotter = VerbAccuracyPlotter(path_config)
    plotter.plot_all()


if __name__ == "__main__":
    main()


