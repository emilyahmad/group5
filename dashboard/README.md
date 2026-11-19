# Pneumonia XAI Dashboard

A clinician-facing explainable-AI dashboard for pneumonia on chest X-rays. It
follows the design references in `../design-references/` and the
"what an effective XAI dashboard presents" checklist in `../CLAUDE.md`.

| Panel | Inspired by | Checklist item |
|---|---|---|
| Worklist (red = High, gray = everything else) | TICICDS C1 | glanceable summary first |
| Low / Medium / High band + guidance, raw score in an expander | TICICDS (no raw scores), Duke (local thresholds) | prediction in context |
| Validate dialog (agree? actions taken? flag) | TICICDS fig. 4 | a way to act and push back |
| X-ray + heatmap (Grad-CAM / Integrated Gradients / Occlusion), lung zones, radiologist outline | Saporta et al. (`../benchmarking.pdf`) | why the model said it |
| Lung-zone contribution bars ("raises / lowers likelihood") | TICICDS fig. 3 | why, in clinical terms |
| Related findings (consolidation, opacity, effusion, …) | AICare B | why / adversarial verification |
| Template summary (no LLM) | AICare C, TICICDS fig. 1 text box | plain language |
| Cohort histogram with this patient marked | AICare D | evidence for trust |
| What the model can't see: OOD check, AP film, off-lung attention, known limits | team notes | what it can't see |
| AUROC, heatmap-vs-radiologist IoU, model + evaluation date | team notes, Duke (local validation) | evidence for trust |

Design rules taken from the papers: categories instead of raw probabilities;
no red/green on values (TICICDS clinicians read it as the model's reasons);
detail behind expanders (progressive disclosure).

Styling uses Stanford identity colours (see `.streamlit/config.toml` and the
palette constants at the top of `app.py`). Cardinal red is the brand colour and
also marks *High likelihood*, which always carries a ▲ icon and the word HIGH.
Orange/blue mean "raises/lowers likelihood" (a pair validated as colour-blind safe).

## Setup

Use the Python environment that has the packages (e.g. Anaconda's
`python3`, not macOS's `/usr/bin/python3`):

```bash
pip install -r dashboard/requirements.txt
```

Model weights (~100 MB in total) download automatically to `~/.torchxrayvision/` on first run.

### Dataset: CheXlocalize (Stanford AIMI)

1. Register (free) and download **CheXlocalize** from the Stanford AIMI shared
   datasets site. You get `CheXlocalize_files.zip` (~3.9 GB).
2. Unzip it into `data/chexlocalize/`. **There are no .jpg or .csv files in it.**
   The X-rays are stored inside `gradcam_maps_val/*.pkl`: each pickle bundles one
   image (stretched to 320×320), its original size, and the radiologist label for
   one finding. Keep these files.
3. Extract the images and labels (about 1 min). The script reads the pickles with a
   restricted loader, since pickles can run code:

   ```bash
   python dashboard/scripts/extract_chexlocalize_pkl.py
   ```

   This writes `data/chexlocalize/CheXpert/val/…/view1_frontal.jpg` (202 frontal
   validation X-rays, restored to their original aspect ratio) and `val_labels.csv`.
4. Score the cohort once (~7 min on a laptop CPU):

   ```bash
   python dashboard/scripts/precompute.py                      # Grad-CAM + IG
   python dashboard/scripts/precompute.py --methods gradcam,ig,occlusion   # slower
   python dashboard/scripts/precompute.py --limit 20           # quick test
   ```

The dataset is git-ignored. Don't commit it.

**Limits of this data**, all shown in the UI:
- Only the 202 frontal **validation** images are available. Test-set images aren't in the release.
- Images are 320 px (fine for the 224 px model, but low-resolution for reading the film).
- **No pneumonia label** and no age, sex or AP/PA. The pneumonia score's AUROC is computed against
  "consolidation **or** lung opacity present" as a labeled proxy.
- **No pneumonia outlines.** The "radiologist outline" is the union of the **Consolidation**
  and **Airspace Opacity** masks.

If you get the full CheXpert images (e.g. Kaggle "CheXpert-v1.0-small" `valid/` + `valid.csv`),
drop them into `data/chexlocalize/` too. The loader picks up those file names as well.

## Run

From the repo root (so `.streamlit/config.toml` is picked up):

```bash
streamlit run dashboard/app.py
```

Without the dataset, the app opens in single-image mode on `data/sample_cxr.png`
and accepts uploads (PNG, JPG or DICOM).

Clinician validations are appended to `dashboard/feedback/validations.csv`.

## Code map

- `app.py`: Streamlit layout
- `xai/model.py`: model loading, preprocessing, shared display frame
- `xai/saliency.py`: Grad-CAM (manual), IG and Occlusion (captum), overlays, IoU
- `xai/regions.py`: PSPNet lung masks, 6 lung zones, zone contributions
- `xai/reliability.py`: autoencoder OOD score, per-image warnings, known limits
- `xai/summary.py`: risk bands, guidance text, template summary
- `xai/dataset.py`: CheXlocalize index and RLE ground-truth masks
- `scripts/precompute.py`: cohort predictions and metrics
- `feedback.py`: validation log

Env overrides: `CHEXLOCALIZE_DIR` (dataset location) and `PNEUMO_CACHE` (cache folder).
