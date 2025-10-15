import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import List
from pathlib import Path

class VerbAccuracyPlotter:
    @dataclass
    class SingleLevelPath:
        human: Path
        llm: Path
        vlm: Path

    def __init__(self, level_name: str, path: SingleLevelPath, output_dir: Path):
        self.level_name = level_name
        self.df_human = pd.read_csv(path.human)
        self.df_llm = pd.read_csv(path.llm)
        self.df_vlm = pd.read_csv(path.vlm)
        self.output_dir = output_dir / level_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbs = sorted(set(self.df_human["Verb"].dropna()))
        self.groups = self._group_verbs(self.verbs)

    def _group_verbs(self, verbs: List[str], group_size: int = 9) -> List[List[str]]:
        return [verbs[i:i + group_size] for i in range(0, len(verbs), group_size)]

    def plot_all(self):
        for idx, group in enumerate(self.groups):
            for source in ["Human", "LLM", "VLM"]:
                self._plot_source_comparison(group, idx, source)

    def _plot_source_comparison(self, verb_group: List[str], group_index: int, source: str):
        df = {"Human": self.df_human, "LLM": self.df_llm, "VLM": self.df_vlm}[source]

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
        ax.set_title(f"{self.level_name.upper()} – {source}: Original vs. Processed (Group {group_index + 1})")
        ax.legend()

        plt.tight_layout()
        out_path = self.output_dir / f"group_{group_index + 1}_{source.lower()}_original_vs_processed.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")

@dataclass
class SingleLevelPath:
    human: Path
    llm: Path
    vlm: Path

@dataclass
class PathConfig:
    lvl3: SingleLevelPath
    indoor_or_outdoor: SingleLevelPath
    manmade_or_natural: SingleLevelPath
    output_path: Path


# Paths setup
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"/"results"/"x_over_20"

def main():
    base = PROJECT_ROOT/DATA_DIR
    config = PathConfig(
        lvl3=SingleLevelPath(
            human=base / "lvl3" / "x_over_20_for_humans.csv",
            llm=base / "lvl3" / "x_over_20_for_llms.csv",
            vlm=base / "lvl3" / "x_over_20_for_vlms.csv"
        ),
        indoor_or_outdoor=SingleLevelPath(
            human=base / "indoor_or_outdoor" / "x_over_20_for_humans.csv",
            llm=base / "indoor_or_outdoor" / "x_over_20_for_llms.csv",
            vlm=base / "indoor_or_outdoor" / "x_over_20_for_vlms.csv"
        ),
        manmade_or_natural=SingleLevelPath(
            human=base / "man_made_or_natural" / "x_over_20_for_humans.csv",
            llm=base / "man_made_or_natural" / "x_over_20_for_llms.csv",
            vlm=base / "man_made_or_natural" / "x_over_20_for_vlms.csv"
        ),
        output_path=base
    )

    for level_name, level_path in {
        "lvl3": config.lvl3,
        "indoor_or_outdoor": config.indoor_or_outdoor,
        "man_made_or_natural": config.manmade_or_natural
    }.items():
        plotter = VerbAccuracyPlotter(level_name, level_path, config.output_path)
        plotter.plot_all()

if __name__ == "__main__":
    main()


