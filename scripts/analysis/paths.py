from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True) #cannot be modified after creation
class PathConfig:
    captions_filepaths: list[Path] | None = None
    finalized_captions_filepaths: list[Path] | None = None
    llm_annotations_filepaths: list[Path] | None = None

    images_filepaths: list[Path] | None = None


    verb_csv_path: Path | None = None
    kld_csv_path: Path | None = None

    # Outputs
    original_js_output_csv: Path | None = None
    finalized_js_output_csv: Path | None = None
    original_captions_grouped_by_verb_csv: Path | None = None
    finalized_captions_grouped_by_verb_csv: Path | None = None
    output_csv_path: Path | None = None


@staticmethod
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@staticmethod
def data_dir() -> Path:
    return PathConfig.project_root() / "data"