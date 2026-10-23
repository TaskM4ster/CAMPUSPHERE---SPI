# CAMPUSPHERE Student Performance Intelligence (SPI)

An independent machine-learning module that flags students who may need support or review, using grades and background information from a public dataset.

The output is a decision-support flag for staff review. It is not a verdict about any student. The module is intended for later integration into the ANHS Portal, but it is developed and evaluated independently first.

## Project structure

```
campusphere-spi/
├── datasets/
│   ├── raw/                  # Original UCI files (do not modify)
│   │   ├── student-mat.csv
│   │   └── student-por.csv
│   └── processed/            # Generated data (not committed)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── preprocessing.py      # Loads data, defines risk label and preprocessing
│   ├── train.py              # Trains and saves Model A
│   └── predict.py            # Scores students with the saved model
├── api/
│   └── main.py               # FastAPI prediction service
├── models/                   # Saved models (generated, not committed)
├── reports/
│   └── figures/              # Charts for the paper
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

UCI Student Performance dataset, Mathematics file (`student-mat.csv`): 395 students, 33 columns. The Portuguese file (`student-por.csv`, 649 students) is used only for external validation. Source, license (CC BY 4.0), citation, and checksums are in `datasets/README.md`.

**Data integrity:** an earlier version of `student-mat.csv` was a synthetic file that imitated the UCI data. It was detected during verification, replaced with the authentic UCI file, and all results from it were discarded. Both files' SHA-256 checksums are recorded in `datasets/README.md`.

## Risk label

The label is derived from the final grade, G3:

| G3       | Label          | Students (Math) |
| -------- | -------------- | --------------- |
| 0 to 9   | `at_risk`      | 130             |
| 10 to 12 | `needs_review` | 134             |
| 13 to 20 | `on_track`     | 131             |

G3 is never used as an input feature, because it defines the label.

## Models

- **Model A (early flag, deployed):** uses information available after period 1. Inputs are G1, failures, studytime, absences, and the background columns. G2 is excluded.
- **Model B (comparison only):** adds G2. It shows how much accuracy is gained by waiting for the second-period grade. It is not deployed.

Both models use standardized numeric features, one-hot encoded text features, and logistic regression with balanced class weights. Both use the same stratified 80/20 split with `random_state=42`.

**Two-tier flag:** a student is flagged for support or review if the model predicts `at_risk` or `needs_review`.

## Methodology notes

- Numeric features are standardized with `StandardScaler`. An earlier unscaled version failed to converge, so its results were replaced.
- The encoder and scaler are fitted on the training rows only, inside the pipeline.
- Cross-validation on the training set is the primary basis for model comparison. The test set was used for reporting, not for choosing models.

## Results

Cross-validation on the training set (5-fold, 316 students):

| Model   | Accuracy       | Macro F1       |
| ------- | -------------- | -------------- |
| Model A | 0.693 (±0.077) | 0.690 (±0.083) |
| Model B | 0.810 (±0.041) | 0.811 (±0.042) |

Test set (79 students):

| Model                               | Accuracy | Macro F1 |
| ----------------------------------- | -------- | -------- |
| Baseline (always most common class) | 0.342    | 0.170    |
| Model A                             | 0.671    | 0.670    |
| Model B                             | 0.785    | 0.783    |

Model A two-tier flag on the test set: recall 0.849, precision 0.882, 51 of 79 students flagged.

External validation (Model A trained on Mathematics, scored on Portuguese students not present in the Mathematics file): 374 unique students were removed as overlapping, leaving 275. Macro F1 is 0.653 (baseline 0.203). The two-tier flag has recall 0.932 and precision 0.868, flagging 204 of 275 students. The pre-set criterion (two-tier recall at least 0.80 and macro F1 above baseline) was met.

**Note on the test set:** it has been used for several reported comparisons, including the baseline, Model B, and the final Model A evaluation, and it was rerun after scaling was added. Cross-validation results are the primary evidence for model choices.

## Setup

Requires **Python 3.12**. From the project folder, in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run order

The model files are not stored in Git, so generate them before starting the API:

```powershell
python src\train.py
uvicorn api.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` to test the API.

## Tech stack (versions used)

- Python 3.12.10
- pandas 3.1.0, NumPy 2.5.4
- scikit-learn 1.9.1
- Matplotlib 3.11.2, Seaborn 0.13.2
- joblib 1.6.0
- FastAPI 0.159.1, Uvicorn 0.54.0
- JupyterLab 4.6.4

## Known limitations

- 38 students have a final grade of 0. Thirteen of them also have G2 = 0, which suggests they left before the second period. They may have withdrawn rather than performed poorly. They are kept in the `at_risk` class. A sensitivity analysis excluding them is planned and not yet reported.
- Attendance is a single total of absences, not per-subject or per-week data. Absences do not differ significantly across risk classes (Kruskal-Wallis p = 0.55).
- G2 is strongly related to G3. Model B's gain over Model A may partly reflect students who left, not only weaker students.
- The data covers two schools and one academic year. Both files come from the same schools, so the external test checks transfer across subjects, not across schools or cohorts.
- The test set has 79 students, so test scores can shift by several points with a different split.

## Responsible use

- Flags are for staff review only. They must not be used for automatic decisions about students.
- The model uses only public, anonymized data. Do not add identifiable ANHS student records to this repository.
- The API stores no student data.
