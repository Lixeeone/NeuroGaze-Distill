# NeuroGaze-Distill
EEG-to-face emotion recognition distillation project
（Note: Despite the name NeuroGaze-Distill, this project does not rely on eye-tracking (gaze) data. “Gaze” denotes the model’s attention-like behavior induced by prototypes and the D-Geo prior.）
# NeuroGaze-Distill (Minimal yet Reproducible)

A compact, reviewer-friendly release of our FER distillation project.
It focuses on the **main model** used in the paper and provides everything needed to **verify numbers** and, if desired, **recompute metrics** on your local data.

---

## Highlights

* **Main model**: `A3_full` (student distilled from teacher), **D-Geo prior enabled**, tuned at **weight = 0.012**, trained **100 epochs**.
* **Prototype knowledge**: fixed **v4** prototypes
  `data/processed/prototypes_dreamer_mahnob_5x5_v4.npz` (DREAMER + MAHNOB static integration).
* **Cross-dataset protocol**:

  * **8-way**: fixed FER mapping (8 classes).
  * **Present-only**: macro-F1 over labels **present in the target dataset**.
* **Metrics**: `acc`, `macro_f1`, `bACC`; saved as `metrics.json` per dataset.
* **Repro convenience**: ready-to-compile LaTeX tables for Overleaf (`viz/*.tex`), eval scripts for **CSV** and **NPZ** manifests, and configuration **fingerprints** (SHA-256) for transparency.

> We **do not** distribute datasets. Model weights and evaluation artifacts are provided. If you have the datasets locally, the included scripts will reproduce the tables.

---

## Repository Layout (what you actually need)

```
.
├─ tools/
│   ├─ eval_student_csv.py      # Evaluate from CSV (CK+ / AffectNet-mini / FERPlus CSV)
│   └─ eval_student_npz.py      # Evaluate from NPZ (FERPlus valid/test) – no filepaths needed
├─ configs/
│   ├─ student.yaml             # Student backbone & dims (must match training)
│   ├─ distill.yaml             # Distillation knobs (kd_mse / proto_kd / d_geo / ce / optim)
│   └─ ablations/
│       └─ A3_full.yaml         # Main model config (100-epoch long-train variant)
├─ outs/
│   ├─ abla_A3_full_100/
│   │   ├─ student_best.ckpt                # Main model weights (100 epochs)
│   │   └─ ablation_fingerprint.json        # Switches + prototype SHA256, for audit
│   ├─ xval_ckplus_abla_A3_full/metrics.json
│   ├─ xval_affmini_abla_A3_full/metrics.json
│   └─ xval_ferplus_valid_abla_A3_full/metrics.json
├─ viz/
│   ├─ xval_main.tex                        # Cross-dataset table (A3_full)
│   └─ ablation_lite.tex                    # Lightweight ablation on FER valid (8-way)
├─ env_conda.yml        # (optional) conda export snapshot
├─ pip_freeze.txt       # pip snapshot (always present)
└─ git_commit.txt       # code snapshot (hash or "unknown")
```

---

## Quickstart (verify without any data)

You can audit the exact switches and produce the same paper tables **without** local datasets.

```bash
# 1) Inspect the training/eval fingerprint
cat outs/abla_A3_full_100/ablation_fingerprint.json

# 2) Compile LaTeX tables directly (or upload to Overleaf)
#    - viz/xval_main.tex     : A3_full cross-dataset table
#    - viz/ablation_lite.tex : FERPlus ablation (8-way)
```

The `metrics.json` under `outs/xval_*_abla_A3_full/` are exactly what those tables read.

---

## Full Evaluation (with your local data)

### Environment

```bash
# Recommended
conda env create -f env_conda.yml -n neurogaze  # if available
conda activate neurogaze

# Fallback (always safe)
pip install -r pip_freeze.txt
```

### Files you need to provide

* **FERPlus (preferred: NPZ)**
  `data/FERPlus/ferplus_valid.npz` and `data/FERPlus/class_map_ferplus.json`
* **CK+ & AffectNet-mini (CSV)**

  ```
  data/xval_manifests/ckplus/labels_all.csv
  data/xval_manifests/affmini/labels_all.csv
  data/xval_manifests/class_map_fer.json
  ```

  CSV must contain **resolvable image paths** and any of these header names:

  * path column: `image_path` / `path` / `img` / `file` / `filepath` / `filename`
  * label name column: `label_name` / `class_name` / `label_str`
  * or label id column: `label_id` / `label` / `cls` / `class` / `target` / `y`

> If training from scratch, ensure prototypes v4 exist at
> `data/processed/prototypes_dreamer_mahnob_5x5_v4.npz`.
> For **evaluation only**, you only need the ckpt and your data.

### Run evaluations

**FERPlus (NPZ, no filepaths needed):**

```bash
PYTHONPATH=. python tools/eval_student_npz.py \
  --ckpt outs/abla_A3_full_100/student_best.ckpt \
  --npz  data/FERPlus/ferplus_valid.npz \
  --class-map data/FERPlus/class_map_ferplus.json \
  --outs-dir outs/xval_ferplus_valid_abla_A3_full
```

**CK+ (CSV):**

```bash
PYTHONPATH=. python tools/eval_student_csv.py \
  --ckpt outs/abla_A3_full_100/student_best.ckpt \
  --csv  data/xval_manifests/ckplus/labels_all.csv \
  --class-map data/xval_manifests/class_map_fer.json \
  --img-size 224 \
  --outs-dir outs/xval_ckplus_abla_A3_full
```

**AffectNet-mini (CSV):**

```bash
PYTHONPATH=. python tools/eval_student_csv.py \
  --ckpt outs/abla_A3_full_100/student_best.ckpt \
  --csv  data/xval_manifests/affmini/labels_all.csv \
  --class-map data/xval_manifests/class_map_fer.json \
  --img-size 224 \
  --outs-dir outs/xval_affmini_abla_A3_full
```

Outputs:

* `outs/.../metrics.json` with:

  ```json
  {
    "8way": {"acc": ..., "macro_f1": ..., "bacc": ...},
    "present_only": {
      "acc": ..., "macro_f1": ..., "bacc": ...,
      "present_idx": [...],
      "present_names": [...]
    }
  }
  ```
* Optional confusion matrices: `confmat.png`, `confmat_present.png` (if plotting deps available).

---

## How we choose the main model (context for reviewers)

We sweep **D-Geo** weights with `tools/sweep_dgeo.sh` and **select by external present-only macro-F1 mean** (CK+ & Affect-mini).
The best setting was **D-Geo = 0.012** with **positive classes** = `["happiness","surprise"]`.
We then **long-train 100 epochs** and set the canonical alias:

```
outs/abla_A3_full    -> outs/abla_A3_full_100
runs/abla_A3_full    -> runs/abla_A3_full_100
```

so all scripts & tables resolve to this “main” variant.

> For speed we also provide `tools/eval_student_npz.py` to avoid any path headaches on FERPlus.

---

## Regenerating paper tables (if you re-run eval)

After you run evaluation and produce `metrics.json`, re-emit the LaTeX:

* `viz/xval_main.tex` — **Cross-dataset** table for `A3_full`.
* `viz/ablation_lite.tex` — **Lightweight ablation** on FER valid (8-way only).

These compile directly in Overleaf.

---

## Training (if you want to retrain)

We keep training instructions minimal (not required for verification):

```bash
# Main model config (100-epoch)
python train_student_distill.py \
  --cfg configs/ablations/A3_full.yaml \
  --workers 8 --channels-last --acc-steps 1 --clip-grad 1.0 \
  --best-metric macro_f1 --runs-dir runs/abla_A3_full_100 --outs-dir outs/abla_A3_full_100 \
  --no-ldacc
```

**Important knobs in `distill.yaml` / `A3_full.yaml`:**

* `kd_mse.weight = 0.30`
* `proto_kd = {mode: cos, weight: 0.12, tau: 0.90, temperature: 5.0}`
* `d_geo = {weight: 0.012, positive_classes: ["happiness","surprise"], schedule: "late_cosine", start_epoch: 20, end_epoch: 60, hard_gate: 0.2, intra_class: true}`
* `ce = {label_smoothing: 0.055, class_weights: [...]}`
* `optim.epochs = 100` (main model); eval temperature `= 1.0`.

**Prototypes v4** are fixed via
`teacher_prototypes: data/processed/prototypes_dreamer_mahnob_5x5_v4.npz`.

---

## Reproducibility & Fingerprints

* `outs/abla_A3_full_100/ablation_fingerprint.json` records:

  * **config hashes** (SHA-256) for `A3_full.yaml`, `student.yaml`
  * **prototype path & SHA-256** (v4)
  * all **distillation switches** & weights (`kd_mse / proto_kd / d_geo`)
  * training epochs and core options
* `env_conda.yml`, `pip_freeze.txt` — environment snapshots
* `git_commit.txt` — code snapshot (hash or “unknown”)

---

## Troubleshooting

* **`rows=0` in CSV eval**: image paths in CSV are not resolvable. Use **absolute paths** or place images exactly where CSV points.
* **Mapping differences**: CK+ / Affect-mini use `class_map_fer.json` (fixed FER 8-way). FERPlus uses `class_map_ferplus.json`.
* **File not found**: double-check `affmini` vs `affectmini` spelling and the paths listed above.
* **GPU/Speed**: You can lower `--workers` for stability, or raise for speed. We default to safe values.

---

## Citation

If you use this repo, please cite the paper (TBD):

```bibtex
@inproceedings{neurogaze_distill_2025,
  title={NeuroGaze-Distill: ...},
  author={...},
  booktitle={...},
  year={2025}
}
```

---

## License

TBD (research-only by default unless specified).

---

## Contact

Open a GitHub Issue (preferred during review) or email **tzulamlee@gmail.com**.  
We’re happy to help reproduce the reported numbers on your environment.

---

<p align="center">
  <a href="https://github.com/user-attachments/assets/c41b87b8-c2ad-457a-9eb2-d6374cd3c85c" target="_blank" rel="noopener">
    <img src="https://github.com/user-attachments/assets/c41b87b8-c2ad-457a-9eb2-d6374cd3c85c"
         alt="NeuroGaze-Distill — overview figure" width="680">
  </a>
  <br/>

</p>


