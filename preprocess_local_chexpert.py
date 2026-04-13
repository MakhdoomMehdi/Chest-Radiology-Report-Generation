"""
preprocess_local.py
═══════════════════
Preprocessing pipeline for MIMIC-CXR chest X-ray / report pairs.

VIRTUAL ENV SETUP (run once before this script):
    python -m venv venv
    venv\\Scripts\\activate          # Windows
    pip install pillow tqdm

USAGE:
    python preprocess_local.py

OUTPUT:
    <OUT_DIR>/
        images/          ← resized JPEGs, one per study
        train.json       ← list of {subject_id, study_id, image, report, chexpert}
        val.json         ← same format, held-out split chexpert = list of 14 ints {0,1} (CheXpert-14 order)
    <OUT_DIR>.zip        ← single archive ready to upload to cloud storage

CheXpert labels come from mimic-cxr-2.0.0-chexpert.csv (MIMIC-CXR-JPG metadata).
Set CHEXPERT_CSV below; if the file is missing, all-zero labels are used and a
warning is printed (baseline xray_report.ipynb still works).

═══════════════════════════════════════════════════════════════════════════════
DATASET STRUCTURE (what this script is designed around)
═══════════════════════════════════════════════════════════════════════════════

The MIMIC-CXR CSV has ONE ROW PER SUBJECT (patient).
Each subject has MULTIPLE STUDIES (clinical visits), and each study has
multiple images (different projection angles: PA, AP, Lateral, LL).

CSV column layout:
  subject_id   patient identifier
  image        Python-list-string of ALL image paths for this subject
               paths follow: files/p<shard>/p<subject_id>/s<study_id>/<img>.jpg
  view         list of all view types present  (PA, AP, LATERAL, LL, nan)
  PA           subset of image paths that are PA (posterior-anterior) views
  AP           subset that are AP (anterior-posterior) views
  Lateral      subset that are lateral views
  text         list of RADIOLOGY REPORTS — ONE PER STUDY, in the same order
               as the unique study IDs that appear in the image list
  text_augment noisy machine-translated version (not used here)

STUDY-TO-REPORT MAPPING (verified by inspection):
  1. Extract unique study IDs from image list in order of first appearance.
     Studies appear in ascending chronological order (s-IDs are monotonic).
  2. text[0] → studies[0], text[1] → studies[1], ...  (perfect 1:1 match)

FRONTAL IMAGE SELECTION per study (PA > AP > any):
  PA views are the clinical standard (patient standing, standard distance).
  AP is used when the patient cannot stand; acceptable fallback.
  We never use lateral-only studies — not enough signal for report generation.

This gives ~150K–200K (study, frontal_image, report) training triples from
the ~64K subject rows in the train CSV.
═══════════════════════════════════════════════════════════════════════════════
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 · CONFIGURATION
# Adjust these before running.
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

# Root of the downloaded kagglehub dataset
DATASET_ROOT = Path(r"D:\kagglehub_cache\datasets\simhadrisadaram\mimic-cxr-dataset\versions\2")

# Subfolder that contains the images (relative paths in the CSV start here)
IMAGE_BASE   = DATASET_ROOT / "official_data_iccv_final"

# Where to write output images and JSON manifests
OUT_DIR      = Path(r"D:\Vision2\preprocessed_cxr")

# CheXpert-14 CSV (PhysioNet MIMIC-CXR-JPG). Filename may differ by release.
CHEXPERT_CSV = DATASET_ROOT / "mimic-cxr-2.0.0-chexpert.csv"

# Column order must match xray_chexpert_aux.ipynb / CheXpert-14 standard.
CHEXPERT_LABEL_COLS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]

# ── Image settings ────────────────────────────────────────────────────────────
# 336×336 is LLaVA-1.5's native CLIP input resolution.
# Using the exact native size avoids a secondary interpolation step inside the
# model's vision encoder, preserving spatial detail in the chest anatomy.
TARGET_SIZE  = (336, 336)

# Quality 82 gives a good sharpness/filesize trade-off for medical images.
# At this size and quality each image is ~25-40 KB, so 150K images ≈ 4-6 GB.
JPG_QUALITY  = 82

# Thread count for parallel resizing.  Set to os.cpu_count() or lower.
NUM_WORKERS  = 8

# Minimum report character length to accept.  Shorter strings are usually
# residual augmentation artefacts (e.g. "Impression: " with nothing after).
MIN_REPORT_LEN = 40

# Set an integer to cap training samples (e.g. 5000 for a quick test run).
# None = use all available data.
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES   = None

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 · IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import ast
import csv
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

print("Python       :", sys.version.split()[0])
print("Pillow       :", Image.__version__)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 · DIRECTORY SETUP
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "images").mkdir(exist_ok=True)

print("\nOutput directory:", OUT_DIR)
print("Image base      :", IMAGE_BASE)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 · PARSING UTILITIES
# These functions handle the raw CSV format:
#   - list-like strings are eval'd safely
#   - study IDs are extracted from image path segments
#   - frontal image selection is PA-first
# ─────────────────────────────────────────────────────────────────────────────

def parse_list(value: str) -> list:
    """
    Safely deserialise a Python list literal stored inside a CSV cell.
    Returns [] for empty, null, or malformed values rather than raising.
    """
    if not value or str(value).strip() in ("", "[]", "nan"):
        return []
    try:
        result = ast.literal_eval(value)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def extract_study_id(image_path: str) -> str | None:
    """
    Pull the study segment (e.g. 's50414267') out of a path like:
        files/p10/p10000032/s50414267/02aa804e-bde0afdd-....jpg
    Returns None if no valid study segment is found.
    """
    for part in Path(image_path).parts:
        if part.startswith("s") and part[1:].isdigit():
            return part
    return None


def get_frontal_for_study(study_id: str, pa_list: list, ap_list: list, all_list: list) -> str | None:
    """
    Select the single best frontal image for a given study.

    Priority: PA > AP > any image from that study.
    We intentionally avoid lateral-only studies because a lateral view alone
    does not contain enough frontal chest anatomy for holistic report generation.
    Returns the relative image path or None if nothing suitable is found.
    """
    tag = f"/{study_id}/"

    pa_match  = [x for x in pa_list  if tag in x]
    ap_match  = [x for x in ap_list  if tag in x]
    any_match = [x for x in all_list if tag in x]

    candidate = pa_match or ap_match or any_match
    return candidate[0] if candidate else None


def get_valid_report(text_value: str, min_len: int = MIN_REPORT_LEN) -> str | None:
    """
    Return the first non-trivial report string from the cell value.
    'Non-trivial' means it contains at least `min_len` characters of content.
    """
    for t in parse_list(text_value):
        t = t.strip()
        if len(t) >= min_len:
            return t
    return None


def normalize_chexpert_study_id(raw: str | int) -> str | None:
    """Match study IDs to path form 's50414267'."""
    s = str(raw).strip()
    if s.startswith("s") and len(s) > 1 and s[1:].isdigit():
        return s
    if s.isdigit():
        return f"s{s}"
    return None


def load_chexpert_lookup(csv_path: Path) -> dict[tuple[str, str], list[int]]:
    """
    Build (subject_id, study_id) ->14 binary labels from the CheXpert CSV.
    Binarization: 1.0 -> 1, 0.0 -> 0, -1.0 (uncertain) -> 1, blank/NaN -> 0.
    """
    lookup: dict[tuple[str, str], list[int]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("subject_id", "")).strip()
            study_norm = normalize_chexpert_study_id(row.get("study_id", ""))
            if not sid or study_norm is None:
                continue
            labels: list[int] = []
            for col in CHEXPERT_LABEL_COLS:
                cell = row.get(col, "")
                if cell is None or str(cell).strip() in ("", "nan", "NaN", "None"):
                    labels.append(0)
                    continue
                try:
                    v = float(str(cell).strip())
                except ValueError:
                    labels.append(0)
                    continue
                if v == -1.0:
                    labels.append(1)
                else:
                    labels.append(1 if v >= 1.0 else 0)
            if len(labels) == 14:
                lookup[(sid, study_norm)] = labels
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 · STUDY-LEVEL RECORD EXTRACTION
#
# This is the core logic that converts the subject-level CSV into a flat list
# of (study, frontal_image_path, report) triples.
#
# For each subject row:
#   1. Parse the image list and extract unique study IDs in order.
#   2. Parse the text list — it has exactly one entry per unique study.
#   3. For each (study_index → study_id), pair it with text[study_index].
#   4. Find the best frontal image for that study.
#   5. Verify the image file actually exists on disk.
#   6. Add the triple to the output list, or count the failure reason.
# ─────────────────────────────────────────────────────────────────────────────

def extract_study_records(
    csv_path: Path,
    split_name: str,
    chexpert_lookup: dict[tuple[str, str], list[int]],
    max_samples: int = None,
) -> list:
    """
    Read a subject-level MIMIC-CXR CSV and return a flat list of study records.
    Each record: {subject_id, study_id, src, dst, report, chexpert}.
    """
    records = []
    chexpert_missing = 0
    skipped = {
        "no_frontal":   0,   # study has no usable frontal image in the CSV lists
        "no_report":    0,   # report string is missing or too short
        "file_missing": 0,   # image path referenced in CSV but not on disk
        "count_mismatch": 0, # study count ≠ report count for this subject (rare)
    }

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_samples and len(records) >= max_samples:
                break

            # ── Parse raw lists ────────────────────────────────────────────────
            all_imgs  = parse_list(row["image"])
            pa_imgs   = parse_list(row["PA"])
            ap_imgs   = parse_list(row["AP"])
            # text: one entry per study, in the same chronological order
            texts     = parse_list(row["text"])

            # ── Extract ordered unique studies ─────────────────────────────────
            # dict.fromkeys preserves insertion order and removes duplicates.
            studies = list(dict.fromkeys(
                extract_study_id(p) for p in all_imgs if extract_study_id(p)
            ))

            # Sanity check: study count should match report count
            if len(studies) != len(texts):
                skipped["count_mismatch"] += len(studies)
                continue

            # ── Build one record per study ─────────────────────────────────────
            for study_idx, study_id in enumerate(studies):

                report = texts[study_idx].strip() if study_idx < len(texts) else None

                # Skip trivially short reports (augmentation artefacts)
                if not report or len(report) < MIN_REPORT_LEN:
                    skipped["no_report"] += 1
                    continue

                # Select the best frontal image for this specific study
                frontal_rel = get_frontal_for_study(study_id, pa_imgs, ap_imgs, all_imgs)
                if not frontal_rel:
                    skipped["no_frontal"] += 1
                    continue

                # Resolve to an absolute path and verify it exists
                src_abs = IMAGE_BASE / frontal_rel
                if not src_abs.exists():
                    skipped["file_missing"] += 1
                    continue

                # Construct a flat output filename: <subject_id>_<study_id>_<imgname>.jpg
                img_stem = Path(frontal_rel).stem
                fname    = f"{row['subject_id']}_{study_id}_{img_stem}.jpg"

                subj = str(row["subject_id"]).strip()
                ck = (subj, study_id)
                if chexpert_lookup:
                    chexpert_vec = chexpert_lookup.get(ck)
                    if chexpert_vec is None:
                        chexpert_missing += 1
                        chexpert_vec = [0] * 14
                else:
                    chexpert_vec = [0] * 14

                records.append({
                    "subject_id": row["subject_id"],
                    "study_id":   study_id,
                    "src":        str(src_abs),
                    "dst":        f"images/{fname}",   # relative path inside OUT_DIR
                    "report":     report,
                    "chexpert":   chexpert_vec,
                })

    print(f"\n[{split_name}]")
    print(f"  Valid study-report pairs : {len(records):>8,}")
    print(f"  Skipped → no_frontal     : {skipped['no_frontal']:>8,}")
    print(f"  Skipped → no_report      : {skipped['no_report']:>8,}")
    print(f"  Skipped → file_missing   : {skipped['file_missing']:>8,}")
    print(f"  Skipped → count_mismatch : {skipped['count_mismatch']:>8,}")
    if chexpert_lookup:
        print(f"  CheXpert rows w/o label : {chexpert_missing:>8,}")
    return records


print("\n" + "─"*60)
print("STEP 1/4  Extracting study-level records from CSVs")
print("─"*60)

if CHEXPERT_CSV.exists():
    print(f"Loading CheXpert labels from:\n  {CHEXPERT_CSV}")
    chexpert_lookup = load_chexpert_lookup(CHEXPERT_CSV)
    print(f"  CheXpert lookup entries: {len(chexpert_lookup):,}")
else:
    chexpert_lookup = {}
    print(
        f"WARNING: CheXpert CSV not found:\n  {CHEXPERT_CSV}\n"
        f"  Using all-zero chexpert vectors. Add the file for xray_chexpert_aux.ipynb."
    )

train_records = extract_study_records(
    DATASET_ROOT / "mimic_cxr_aug_train.csv", "train", chexpert_lookup, MAX_TRAIN_SAMPLES
)
val_records = extract_study_records(
    DATASET_ROOT / "mimic_cxr_aug_validate.csv", "val", chexpert_lookup, MAX_VAL_SAMPLES
)

print(f"\n  Total training pairs : {len(train_records):,}")
print(f"  Total val pairs      : {len(val_records):,}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 · PARALLEL IMAGE RESIZING
#
# Each image is opened, converted to RGB (handles grayscale DICOM-derived JPEGs
# which are L-mode or LA-mode), resized to TARGET_SIZE, and saved as JPEG.
#
# ThreadPoolExecutor is appropriate here because Pillow releases Python's GIL
# during image decode and JPEG encode, so threads genuinely run in parallel.
#
# A progress bar updates per completed future (not per submission) so the
# reported throughput reflects actual disk I/O speed.
# ─────────────────────────────────────────────────────────────────────────────

def resize_one(rec: dict) -> tuple[dict | None, str | None]:
    """
    Resize a single image and return (output_record, error_message).
    Output record replaces the 'src' field with 'image' (relative path).
    """
    try:
        img = Image.open(rec["src"]).convert("RGB")
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        img.save(OUT_DIR / rec["dst"], "JPEG", quality=JPG_QUALITY, optimize=True)
        return {
            "subject_id": rec["subject_id"],
            "study_id":   rec["study_id"],
            "image":      rec["dst"],
            "report":     rec["report"],
            "chexpert":   rec["chexpert"],
        }, None
    except Exception as e:
        return None, f"{rec['src']}: {e}"


def materialize(records: list, split_name: str) -> list:
    """
    Process all records in parallel and return the list of saved output dicts.
    """
    out, errors = [], []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(resize_one, r): r for r in records}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"  Resizing {split_name}", unit="img"):
            result, err = future.result()
            if result:
                out.append(result)
            else:
                errors.append(err)

    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors[:5]:
            print(f"    {e}")
        if len(errors) > 5:
            print(f"    ... and {len(errors)-5} more")

    return out


print("\n" + "─"*60)
print("STEP 2/4  Resizing images (parallel, target: 336×336)")
print("─"*60)

train_data = materialize(train_records, "train")
val_data   = materialize(val_records,   "val")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 · WRITE JSON MANIFESTS
#
# train.json and val.json are flat lists of records.
# Each record: {subject_id, study_id, image (relative path), report, chexpert}.
# The image path is relative to OUT_DIR so it remains valid after the zip is
# extracted in any environment regardless of absolute path differences.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*60)
print("STEP 3/4  Writing JSON manifests")
print("─"*60)

(OUT_DIR / "train.json").write_text(
    json.dumps(train_data, ensure_ascii=False, indent=2), encoding="utf-8"
)
(OUT_DIR / "val.json").write_text(
    json.dumps(val_data, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"  train.json  : {len(train_data):,} records")
print(f"  val.json    : {len(val_data):,} records")

# Sample preview so you can visually verify the output
sample = train_data[0]
print(f"\n  Sample record:")
print(f"    subject_id : {sample['subject_id']}")
print(f"    study_id   : {sample['study_id']}")
print(f"    image      : {sample['image']}")
print(f"    report     : {sample['report'][:120]}...")
print(f"    chexpert   : {sample['chexpert']}  (sum={sum(sample['chexpert'])})")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 · SIZE ESTIMATION + ARCHIVE CREATION
#
# Measure the total size of the output folder before archiving.
# Warn loudly if it approaches the 15 GB Google Drive free-tier limit.
# The zip archive is placed one directory level above OUT_DIR so it does not
# get included in its own contents.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*60)
print("STEP 4/4  Packaging archive")
print("─"*60)

def folder_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9

raw_size = folder_size_gb(OUT_DIR)
print(f"  Output folder size  : {raw_size:.2f} GB (before compression)")

DRIVE_LIMIT_GB = 14.0   # leave 1 GB headroom from the 15 GB free tier
if raw_size > DRIVE_LIMIT_GB:
    print(f"\n  WARNING: folder is {raw_size:.1f} GB — likely exceeds Drive free tier after zipping.")
    print("  Consider reducing MAX_TRAIN_SAMPLES or lowering JPG_QUALITY.")
    print("  Proceeding with archive creation anyway...")

print("  Creating zip archive (this may take several minutes)...")
shutil.make_archive(str(OUT_DIR), "zip", str(OUT_DIR))

zip_path    = Path(f"{OUT_DIR}.zip")
zip_size_gb = zip_path.stat().st_size / 1e9
print(f"  Archive size        : {zip_size_gb:.2f} GB")
print(f"  Archive path        : {zip_path}")

if zip_size_gb > DRIVE_LIMIT_GB:
    print(f"\n  WARNING: zip is {zip_size_gb:.1f} GB — too large for a single Drive upload.")
    print("  Re-run with MAX_TRAIN_SAMPLES set to a smaller value, then upload in batches.")
else:
    print("\n  Archive fits within Google Drive free tier — ready to upload.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 · FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*60)
print("PREPROCESSING COMPLETE")
print("═"*60)
print(f"  Training samples    : {len(train_data):,}")
print(f"  Validation samples  : {len(val_data):,}")
print(f"  Image resolution    : {TARGET_SIZE[0]}×{TARGET_SIZE[1]} px")
print(f"  JPEG quality        : {JPG_QUALITY}")
print(f"  Output folder       : {OUT_DIR}")
print(f"  Archive to upload   : {zip_path}")
print()
print("NEXT STEP:")
print(f"  Upload {zip_path.name} to the root of your Google Drive,")
print("  then open xray_report.ipynb or xray_chexpert_aux.ipynb from Step 1 onwards.")
print("═"*60)
