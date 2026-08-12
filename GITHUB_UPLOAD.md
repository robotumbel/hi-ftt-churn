# GitHub upload metadata (fill these in the web UI)

## New-repository fields

- **Repository name:** `hi-ftt-churn`
- **Description (the "About" box):**
  > Hospitality-Informed Feature-Tokenizer Transformer (HI-FTT) for interpretable hotel booking-cancellation churn prediction — code, data splits, and prediction arrays for reproducibility.
- **Visibility:** Public
- **Do NOT** initialise with README/License/.gitignore (they are already in this package).

## Topics (Settings -> Topics, or the "About" gear -> Topics)

```
churn-prediction  transformer  ft-transformer  tabular-deep-learning
interpretability  hotel-cancellation  hospitality  crm  pytorch
delong-test  reproducibility
```

## Website field (optional)

Leave blank during review, or point to the anonymised mirror (below).

---

## Blind-review warning (read before uploading)

The manuscript is under **double-blind** review. Uploading to a GitHub account
that carries your real name/username **de-anonymises** the submission.

- **During review:** use an anonymised mirror instead — upload
  `hi-ftt-churn-release.zip` to <https://anonymous.4open.science> (or point it at a
  private repo). Put the resulting anonymous URL in the manuscript's
  *Data and code availability* section (currently a placeholder).
- **On acceptance:** create the public GitHub repo, replace the anonymised author in
  `CITATION.cff` and the `Copyright (c)` line in `LICENSE` with the real author list,
  and swap the manuscript URL to the public one.

---

## Manual upload via the web UI

1. Create the empty repo with the fields above.
2. On the repo page: **Add file -> Upload files**.
3. Drag the *contents* of this folder (not the folder itself): `README.md`,
   `LICENSE`, `CITATION.cff`, `.gitignore`, `.gitattributes`, `requirements.txt`,
   `main.py`, `run_pipeline.sh`, `src/`, `data/`, `reproducibility/`.
4. Commit message: `HI-FTT reproducibility release`.
5. Commit.

## Or push the prepared local repo (keeps the anonymised commit)

```bash
cd "D:/Code-PhD/Churn Prediction/hi-ftt-churn-release"
git branch -M main
git remote add origin https://github.com/<username>/hi-ftt-churn.git
git push -u origin main
```
