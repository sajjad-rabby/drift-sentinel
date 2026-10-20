"""CSV reading, writing and column typing.

Public helpers:
- read_csv / write_csv: lossless round-trip for string data
- split_numeric_categorical: column typing helper
    - read_csv: path -> list of row dicts (float coercion where possible)
    - write_csv: rows -> path
    - split_numeric_categorical: column typing helper
"""

import csv
from typing import Dict, List, Sequence

__all__ = ["read_csv", "write_csv", "split_numeric_categorical"]


def _is_numeric_like(values: Sequence[str]) -> bool:
    """True when every non-empty value parses as a float."""
    return all(
        _parse_float(v) is not None for v in values if v is not None and str(v).strip() != ""
    )


def _parse_float(text: str):
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def read_csv(path: str) -> List[Dict[str, object]]:
    """Read a CSV file into a list of row dictionaries.

Raises:
    ValueError: If the file has no header row.
      - If every non-empty value parses as a float, the column is numeric.
        Empty cells in a numeric column become ``0.0`` (documented sentinel
        so downstream binning never receives bare strings).
      - Otherwise the column is treated as categorical text and empty
        cells remain the empty string.

    Raises ``ValueError`` if the file has no header row.
    """
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("CSV file has no header row: {}".format(path))
        raw_rows = list(reader)

    numeric_columns = {
        name
        for name in fieldnames
        if _is_numeric_like([row.get(name, "") for row in raw_rows])
    }

    rows: List[Dict[str, object]] = []
    for raw in raw_rows:
        row: Dict[str, object] = {}
        for name in fieldnames:
            value = raw.get(name, "")
            if name in numeric_columns:
                if value is None or str(value).strip() == "":
                    row[name] = 0.0
                else:
                    row[name] = float(str(value).strip())
            else:
                row[name] = "" if value is None else str(value)
        rows.append(row)
    return rows


def write_csv(rows: Sequence[Dict[str, object]], path: str) -> None:
    """Write row dictionaries to a CSV file.

Args:
    rows: Rows to write.
    path: Destination path.

Column order is the union of keys in first-seen order.

    Column order is the union of keys in first-seen order.
    """
    if not rows:
        # Write an empty file; there is no header information to infer.
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def split_numeric_categorical(
    rows: Sequence[Dict[str, object]],
) -> Dict[str, List[str]]:
    """Split column names into numeric and categorical groups.

Returns:
    Mapping with ``numeric`` and ``categorical`` lists. A column counts as numeric when
every non-empty value parses as a number, and is categorical otherwise. Returns ``{"numeric": [...], "categorical": [...]}``.

    A column is numeric when every non-empty value is a float/int/bool;
    otherwise (including string-valued columns and all-empty columns) it
    is categorical. Returns ``{"numeric": [...], "categorical": [...]}``.
    """
    numeric: List[str] = []
    categorical: List[str] = []
    if not rows:
        return {"numeric": numeric, "categorical": categorical}
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    for name in fieldnames:
        values = [row.get(name) for row in rows]
        non_empty = [v for v in values if v is not None and str(v).strip() != ""]
        all_numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_empty)
        if non_empty and all_numeric:
            numeric.append(name)
        else:
            categorical.append(name)
    return {"numeric": numeric, "categorical": categorical}
