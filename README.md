# CAMPUSPHERE Student Performance Intelligence (SPI)

An independent machine-learning module that flags students who may need support or review, using grades and background information from a public dataset.

The output is a decision-support flag for staff review. It is not a verdict about any student. The module is intended for later integration into the ANHS Portal, but it is developed and evaluated independently first.

## Project structure

```
campusphere-spi/
├── datasets/
│   ├── raw/                  # Original dataset (do not modify)
│   │   └── student-mat.csv
│   └── processed/            # Generated data (not committed)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── preprocessing.py      # Loads data, defines risk label and encoder
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

UCI Student Performance dataset (Mathematics file): 395 students, 33 columns. Full source, license (CC BY 4.0), citation, and limitations are in `datasets/README.md`.

## Risk label

The label is derived from the final grade, G3:

| G3       | Label          |
| -------- | -------------- |
| 0 to 9   | `at_risk`      |
| 10 to 12 | `needs_review` |
| 13 to 20 | `on_track`     |

G3 is never used as an input feature, because it defines the label.

## Models

- **Model A (early flag, deployed):** uses information available after period 1. Inputs are G1, failures, studytime, absences, and the background columns. G2 is excluded.
- **Model B (comparison only):** adds G2. It shows how much accuracy is gained by waiting for the second-period grade. It is not deployed.

Both models use logistic regression with balanced class weights, and both use the same stratified 80/20 split with `random_state=42`.

**Two-tier flag:** a student is flagged for support or review if the model predicts `at_risk` or `needs_review`.

## Results so far

Cross-validation on the training set (5-fold, 316 students):

| Model   | Accuracy       | Macro F1       |
| ------- | -------------- | -------------- |
| Model A | 0.696 (±0.021) | 0.668 (±0.024) |
| Model B | 0.756 (±0.040) | 0.739 (±0.032) |

Test set (79 students):

| Model                               | Accuracy | Macro F1 |
| ----------------------------------- | -------- | -------- |
| Baseline (always most common class) | 0.506    | 0.224    |
| Model A                             | 0.684    | 0.657    |
| Model B                             | 0.709    | 0.676    |

Model A two-tier flag on the test set: recall 0.891, precision 0.934, 61 of 79 students flagged.

**Note on the test set:** it was used three times: for the baseline comparison, for Model B, and for the final Model A evaluation. Cross-validation results are the primary evidence for model choices.

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

- Four students have a final grade of 0. They may have withdrawn rather than performed poorly. They are kept in the `at_risk` class.
- Attendance is a single total of absences, not per-subject or per-week data. Absences do not differ across risk classes (Kruskal-Wallis p = 0.99).
- The data covers one subject (Mathematics), two schools, and one academic year, so results may not transfer to other settings.
- The test set has 79 students, so test scores can shift by several points with a different split.
- G1 and G2 are strongly related to G3, so the model partly learns from earlier grades.

## Responsible use

- Flags are for staff review only. They must not be used for automatic decisions about students.
- The model uses only public, anonymized data. Do not add identifiable ANHS student records to this repository.
- The API stores no student data.
