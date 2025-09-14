

---

# NeuroGaze-Distill

EEG-to-face emotion recognition distillation project

### *(Note: Despite the name **NeuroGaze-Distill**, this project does **not** rely on eye-tracking (gaze) data. “Gaze” denotes attention-like behavior induced by prototypes and the D-Geo prior.)*

---

## 🔔 For Reviewers — Main Reproducibility Package

**Please go to the GitHub *Releases* section** and download the reproducibility bundle:

**Main reproducibility package (A3\_full, 100ep, v4) — Latest**
Tag: **v2025-09-13**  •  Commit: **c4b56e1**
Assets:

* `NeuroGaze_Distill_main_2025-09-13.tar.gz`
  `sha256: f83031befc3e2ffba9bc0caf9a71ab3913defb170ed6dbe04fac0262a174bd84`
* `SHA256SUMS.txt` (checksums for verification)

**Verify locally:**

```bash
sha256sum -c SHA256SUMS.txt
# expected: NeuroGaze_Distill_main_2025-09-13.tar.gz: OK
```

The package contains the **main model weights**, **metrics**, **LaTeX tables**, and **evaluation scripts** needed to verify the paper **without re-training**.

---

## Minimal Yet Reproducible

This repository focuses on the **main model used in the paper** and what’s needed to (a) **verify the numbers** and (b) **optionally recompute metrics** locally.

### Highlights

* **Main model**: `A3_full` (student distilled from teacher), **D-Geo enabled**, tuned at **weight = 0.012**, **100 epochs**.
* **Prototype knowledge**: fixed **v4** prototypes
  `data/processed/prototypes_dreamer_mahnob_5x5_v4.npz` (DREAMER + MAHNOB).
* **Cross-dataset protocol**:
  • **8-way** (fixed FER mapping)  •  **Present-only** (macro-F1 over labels present in the target dataset)
* **Metrics**: `acc`, `macro_f1`, `bACC`; each dataset emits a `metrics.json`.
* **Repro convenience**: ready-to-compile LaTeX tables (`viz/*.tex`), eval scripts for **CSV**/**NPZ**, and **configuration fingerprints** (SHA-256).

> We **do not distribute datasets**. We provide weights and evaluation artifacts. If you have the datasets locally, the included scripts will reproduce the tables.

---

## What’s in the Release Archive

After extracting `NeuroGaze_Distill_main_2025-09-13.tar.gz` you’ll see (abbreviated):

```
.
├─ tools/
│   ├─ eval_student_csv.py      # Evaluate from CSV (CK+ / AffectNet-mini / FERPlus CSV)
│   └─ eval_student_npz.py      # Evaluate from NPZ (FERPlus valid/test) – no filepaths needed
├─ configs/
│   ├─ student.yaml
│   ├─ distill.yaml
│   └─ ablations/A3_full.yaml   # Main config (100-epoch long-train)
├─ outs/
│   ├─ abla_A3_full_100/
│   │   ├─ student_best.ckpt                # Main model weights (100 epochs)
│   │   └─ ablation_fingerprint.json        # Switches + prototype SHA256 (audit)
│   ├─ xval_ckplus_abla_A3_full/metrics.json
│   ├─ xval_affmini_abla_A3_full/metrics.json
│   └─ xval_ferplus_valid_abla_A3_full/metrics.json
├─ viz/
│   ├─ xval_main.tex                        # Cross-dataset table (A3_full)
│   └─ ablation_lite.tex                    # Lightweight FER ablation (8-way)
├─ env_conda.yml            # (optional) conda snapshot (if available)
├─ pip_freeze.txt           # pip snapshot
└─ git_commit.txt           # code snapshot (hash or "unknown")
```

---

## Quick Verify (No Data Needed)

You can audit the exact switches and produce the same paper tables **without any datasets**:

```bash
# 1) Inspect training/eval fingerprint
cat outs/abla_A3_full_100/ablation_fingerprint.json

# 2) Compile LaTeX tables directly (or copy to Overleaf)
#    - viz/xval_main.tex     : A3_full cross-dataset table
#    - viz/ablation_lite.tex : FERPlus ablation (8-way)
```

The `metrics.json` under `outs/xval_*_abla_A3_full/` are exactly what those tables read.

---

## Full Evaluation (With Your Local Data)

### Environment

```bash
# Recommended (if provided)
conda env create -f env_conda.yml -n neurogaze
conda activate neurogaze

# Fallback (always safe)
pip install -r pip_freeze.txt
```

### Provide These Files

* **FERPlus (preferred: NPZ)**
  `data/FERPlus/ferplus_valid.npz` and `data/FERPlus/class_map_ferplus.json`
* **CK+ & AffectNet-mini (CSV)**

  ```
  data/xval_manifests/ckplus/labels_all.csv
  data/xval_manifests/affmini/labels_all.csv
  data/xval_manifests/class_map_fer.json
  ```

  CSV must contain resolvable **image paths** and any of these headers:

  * path: `image_path` / `path` / `img` / `file` / `filepath` / `filename`
  * label name: `label_name` / `class_name` / `label_str`
  * or label id: `label_id` / `label` / `cls` / `class` / `target` / `y`

> If training from scratch, ensure prototypes v4 exists at
> `data/processed/prototypes_dreamer_mahnob_5x5_v4.npz`.
> For **evaluation only**, the **ckpt + your data** are sufficient.

### Run Evaluations

**FERPlus (NPZ):**

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

**Outputs** (per dataset):

* `outs/.../metrics.json`
* optional `confmat.png` and `confmat_present.png` (if plotting deps available)

---

## How the Main Model Was Chosen (Context)

We swept **D-Geo** with `tools/sweep_dgeo.sh` and selected by **external present-only macro-F1 mean** (CK+ & Affect-mini).
Best: **D-Geo = 0.012**, `positive_classes = ["happiness","surprise"]`.
Then we **long-trained 100 epochs** and aliased:

```
outs/abla_A3_full -> outs/abla_A3_full_100
runs/abla_A3_full -> runs/abla_A3_full_100
```

---

## Regenerate Paper Tables (If You Re-Run Eval)

After generating new `metrics.json`, re-emit:

* `viz/xval_main.tex` — cross-dataset table for `A3_full`
* `viz/ablation_lite.tex` — lightweight FER ablation (8-way)

These compile directly in Overleaf.

---

## (Optional) Training Command

Not required for verification, but provided for completeness:

```bash
python train_student_distill.py \
  --cfg configs/ablations/A3_full.yaml \
  --workers 8 --channels-last --acc-steps 1 --clip-grad 1.0 \
  --best-metric macro_f1 --runs-dir runs/abla_A3_full_100 --outs-dir outs/abla_A3_full_100 \
  --no-ldacc
```

**Key knobs (in `distill.yaml`/`A3_full.yaml`):**

* `kd_mse.weight = 0.30`
* `proto_kd = {mode: cos, weight: 0.12, tau: 0.90, temperature: 5.0}`
* `d_geo = {weight: 0.012, positive_classes: ["happiness","surprise"], schedule: "late_cosine", start_epoch: 20, end_epoch: 60, hard_gate: 0.2, intra_class: true}`
* `ce = {label_smoothing: 0.055, class_weights: [...]}`
* `optim.epochs = 100` (eval temperature = `1.0`)
* Prototypes v4: `teacher_prototypes: data/processed/prototypes_dreamer_mahnob_5x5_v4.npz`

---

## Reproducibility & Fingerprints

* `outs/abla_A3_full_100/ablation_fingerprint.json` records:

  * SHA-256 of **A3\_full.yaml**, **student.yaml**
  * prototype **path & SHA-256** (v4)
  * all distillation switches/weights (`kd_mse` / `proto_kd` / `d_geo`)
  * training epochs and core options
* `env_conda.yml`, `pip_freeze.txt` — environment snapshots
* `git_commit.txt` — code snapshot (hash or “unknown”)

---

## Troubleshooting

* **`rows=0` in CSV eval** → image paths in CSV are not resolvable. Use **absolute paths** or place images exactly where CSV points.
* **Mapping differences** → CK+ / Affect-mini use `class_map_fer.json` (fixed FER 8-way). FERPlus uses `class_map_ferplus.json`.
* **Spelling** → double-check `affmini` vs `affectmini`.
* **GPU/Speed** → adjust `--workers`. We default to safe values.

---

## Ethics & Privacy

**Datasets.** We use only public datasets (CK+, AffectNet/mini, FERPlus/FER2013, DREAMER, MAHNOB-HCI) under their original academic licenses. We **do not collect new human data** and **do not redistribute** any raw images or videos.

**What we release.** Model weights, evaluation scripts, metrics, and LaTeX tables. The teacher “v4 prototypes” are **aggregated, cross-subject feature statistics** (not per-subject samples) and **not intended to be reversible** to any identity.

**Intended use & restrictions.** Research use only. **Explicitly not for identity recognition or surveillance.** Users must obtain the original datasets directly from the official sources and accept their licenses.

**IRB/ethics review.** In most institutions, using licensed public datasets without collecting new human subjects typically qualifies as **Not Human Subjects Research / Exempt**. We will obtain/retain such determination if required. We include this statement to support transparency during peer review.

---

## Citation

If you use this repo, please cite the paper (TBD):

```bibtex
@inproceedings{neurogaze_distill_2025,
  title={NeuroGaze-Distill: ...},
  author={...},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

---

## License

TBD (research-only by default unless specified).

---

* We thank the maintainers of CK+, AffectNet, FERPlus/FER2013, DREAMER, and MAHNOB-HCI for providing the datasets that enabled this research.

---
