from __future__ import annotations
import csv
from importlib.resources import path
import json
from pathlib import Path
from typing import Iterable, Any
import pandas as pd
from torch import obj




def read_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def write_csv_rows(path: Path, rows: Iterable[Iterable[Any]], header: Iterable[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
    if header:
        writer.writerow(list(header))
    for row in rows:
        writer.writerow(list(row))




def write_dict_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def dump_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)