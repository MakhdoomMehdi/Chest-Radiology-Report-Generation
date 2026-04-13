# CheXpert labels: preprocessing notes

[`preprocess_local.py`](preprocess_local.py) is **already updated**: it loads CheXpert-14 labels (stdlib `csv` only, no pandas) and writes `chexpert` on every record.

Use this file as a reference for CSV paths, column order, and troubleshooting.

---

## 1. Dependencies

Same as before: `pillow`, `tqdm`. **Pandas is not required.**

---

## 2. Obtain the CheXpert CSV

You need the official **CheXpert label file** for MIMIC-CXR (often named `mimic-cxr-2.0.0-chexpert.csv` in the PhysioNet / MIMIC-CXR-JPG distribution).

- Place it under your dataset root, e.g.  
  `DATASET_ROOT / "mimic-cxr-2.0.0-chexpert.csv"`
- If your Kaggle bundle uses a **different path or filename**, set `CHEXPERT_CSV` accordingly.

---

## 3. Add config constants (Section 1, near `DATASET_ROOT`)

```python
# CheXpert-14 labels CSV (path must exist on your machine)
CHEXPERT_CSV = DATASET_ROOT / "mimic-cxr-2.0.0-chexpert.csv"

CHEXPERT_LABEL_COLS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]
```

---

## 4. Load lookup table once (after imports, before `extract_study_records`)

```python
import pandas as pd

def load_chexpert_lookup(csv_path: Path) -> dict[tuple[str, str], list[int]]:
    """
    Map (subject_id, study_id) -> list of 14 ints {0,1}.
    study_id in CSV is often numeric; normalize to 's########' to match extract_study_id().
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CheXpert CSV not found: {csv_path}\n"
            "Download MIMIC-CXR CheXpert labels and set CHEXPERT_CSV."
        )
    df = pd.read_csv(csv_path)
    lookup: dict[tuple[str, str], list[int]] = {}
    for _, row in df.iterrows():
        sid = str(row["subject_id"])
        raw_study = row["study_id"]
        study_id = f"s{raw_study}" if not str(raw_study).startswith("s") else str(raw_study)
        labels: list[int] = []
        for col in CHEXPERT_LABEL_COLS:
            val = row.get(col)
            if pd.isna(val):
                labels.append(0)
            elif val == -1.0:
                labels.append(1)  # uncertain → positive (same convention as README_CHEXPERT_AUX)
            else:
                labels.append(int(val))
        lookup[(sid, study_id)] = labels
    return lookup


chexpert_lookup = load_chexpert_lookup(CHEXPERT_CSV)
print(f"CheXpert lookup rows: {len(chexpert_lookup):,}")
```

---

## 5. `extract_study_records`: attach `chexpert` to each record

Inside the loop where you `records.append({...})`, after you have `study_id` and `row["subject_id"]`:

```python
key = (str(row["subject_id"]), study_id)
chexpert_vec = chexpert_lookup.get(key, [0] * 14)

records.append({
    "subject_id": row["subject_id"],
    "study_id":   study_id,
    "src":        str(src_abs),
    "dst":        f"images/{fname}",
    "report":     report,
    "chexpert":   chexpert_vec,
})
```

Optional: increment a counter when `key not in chexpert_lookup` to verify match rate.

---

## 6. `resize_one`: copy `chexpert` into the output JSON record

Change the returned dict to include the vector:

```python
return {
    "subject_id": rec["subject_id"],
    "study_id":   rec["study_id"],
    "image":      rec["dst"],
    "report":     rec["report"],
    "chexpert":   rec["chexpert"],
}, None
```

---

## 7. Update the top-of-file docstring (OUTPUT section)

Document that each JSON row now includes:

```text
chexpert — list of 14 ints {0,1} in CheXpert-14 order
```

---

## 8. Verify output

After running `python preprocess_local.py`, open `train.json` and confirm the first object has:

```json
"chexpert": [0, 1, 0, ...]
```

If every row is all zeros, the `(subject_id, study_id)` keys likely do not match the CSV (check `study_id` formatting: with vs without `s` prefix).

---

## Related files

| File | Role |
|------|------|
| [`README_CHEXPERT_AUX.md`](README_CHEXPERT_AUX.md) | Full CheXpert-aux training README |
| [`xray_chexpert_aux.ipynb`](xray_chexpert_aux.ipynb) | Notebook that consumes `chexpert` |
| [`preprocess_local.py`](preprocess_local.py) | Script to edit using this checklist |

The baseline notebook [`xray_report.ipynb`](xray_report.ipynb) ignores extra JSON keys; only the CheXpert-aux notebook requires `chexpert`.
