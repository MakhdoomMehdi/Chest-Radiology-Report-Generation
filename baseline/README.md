# Parameter-Efficient Fine-Tuning of LLaVA for Chest Radiology Report Generation

Fine-tunes LLaVA-1.5-7B to generate structured radiology reports from chest X-ray images using QLoRA, updating fewer than 1% of model parameters.

---

## Files

| File | Purpose |
|------|---------|
| `preprocess_local.py` | Run on your local machine. Converts the raw dataset into study-level image-report pairs, resizes images, and packages output into a zip. |
| `xray_researcher.ipynb` | Training notebook. Loads preprocessed data, fine-tunes LLaVA with QLoRA, runs inference, saves adapters. |
| `xray.ipynb` | Original working notebook — preserved as-is. |

---

## Dataset Structure

The MIMIC-CXR CSV has **one row per patient subject**. Each subject has multiple **studies** (clinical visits), and each study has multiple images (different X-ray projections). The `text` column holds one radiology report per study, in the same chronological order as the studies appear in the `image` list.

```
Subject 10003502
 ├── Study s50084553  →  1 image (AP)          →  report[0]
 ├── Study s51180958  →  2 images (AP + Lat)   →  report[1]
 ├── Study s52139270  →  2 images (AP + Lat)   →  report[2]
 └── Study s57812613  →  2 images (PA + AP)    →  report[7]
```

`preprocess_local.py` correctly maps each study to its report and selects the best frontal projection (PA → AP → any).

---

## Optimisation Stack

### Memory efficiency

| Technique | Effect |
|-----------|--------|
| 4-bit NF4 quantisation | Model footprint ~14 GB → ~4 GB |
| Double quantisation | Saves ~0.4 bits/parameter on top of NF4 |
| Paged AdamW 8-bit | Optimizer states paged to CPU; prevents OOM spikes |
| Gradient checkpointing | Recomputes activations on backward pass; large activation memory reduction |
| Frozen vision tower | Eliminates gradients for 304M CLIP parameters entirely |

### Training quality

| Technique | Effect |
|-----------|--------|
| QLoRA rank-16 | ~0.5% of parameters updated; avoids catastrophic forgetting |
| NEFTune noise (α=5) | Embedding-space regularisation; improves generation quality at zero cost |
| Group-by-length batching | Minimises padding tokens per batch |
| Cosine LR with 3% warmup | Smooth convergence without early instability |
| Effective batch 16 | Stable gradient signal via 2×8 accumulation |

### Data pipeline

| Technique | Effect |
|-----------|--------|
| Study-level extraction | Expands ~64K subject rows into ~150K+ (image, report) pairs |
| PA → AP → any frontal selection | Consistent high-quality supervision signal |
| Native 336×336 resize | Exact CLIP input resolution; no interpolation artefacts inside the model |
| Parallel resize (ThreadPoolExecutor) | Saturates CPU cores; reduces preprocessing wall time |
| Prompt-token masking (−100) | Loss computed on report text only; model never penalised for reproducing the prompt |

---

## Step-by-Step Workflow

### Phase 1 — Local Preprocessing (`preprocess_local.py`)

**Setup virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install pillow tqdm
```

**Configure paths** in `preprocess_local.py` Section 1:
```python
DATASET_ROOT = Path(r"path\to\mimic-cxr-dataset\versions\2")
OUT_DIR      = Path(r"path\to\output_folder")
```

**Run:**
```bash
python preprocess_local.py
```

The script will:
1. Parse both CSVs and extract one (frontal image, report) pair per study
2. Resize all images to 336×336 in parallel
3. Write `train.json` and `val.json` manifests
4. Package everything into a single zip archive
5. Report the archive size and warn if it exceeds 14 GB

**Upload** the resulting `.zip` to the root of your Google Drive.

---

### Phase 2 — Training (`xray_researcher.ipynb`)

Open the notebook in Google Colab with a GPU runtime, then run cells in order:

| Step | What it does |
|------|-------------|
| 1 | Set all configuration (paths, model, LoRA, training knobs) |
| 2 | Install dependencies |
| 3 | Mount Drive, extract archive, verify data |
| 4 | Load LLaVA-1.5-7B with 4-bit NF4 quantisation |
| 5 | Inject QLoRA adapters, freeze vision tower |
| 6 | Build dataset and collator |
| 7 | Train with auto-resume from latest checkpoint |
| 8 | Save LoRA adapters (~50–100 MB) to Drive |
| 9 | Run qualitative inference on validation samples |
| 10 | Load saved adapters in a future session |

**On session interruption:** reconnect, re-run Steps 1–6, then re-run Step 7. The training cell automatically resumes from the latest checkpoint saved to Drive.

---

## Expected Numbers

| Metric | Value |
|--------|-------|
| Study-level training pairs | ~150K–200K (from ~64K subject rows) |
| Trainable parameters | ~107M of 7B (~1.5%) |
| VRAM usage (T4 16 GB) | ~12–14 GB |
| Archive size | ~4–7 GB |
| Training time (1 epoch) | ~3–4 hours |
| Saved adapter size | ~50–100 MB |

---

## Limitations

- Evaluation is qualitative in this notebook. For publication, compute BLEU-4, ROUGE-L, and CheXbert F1 on the full validation set.
- Only one frontal image per study is used. Incorporating the paired lateral view could improve anatomical coverage.
- The `text_augment` column (machine-translated paraphrases) is not used. It could serve as contrastive negatives or data augmentation in future experiments.
