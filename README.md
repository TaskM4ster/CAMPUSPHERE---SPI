# CAMPUSPHERE Student Performance Intelligence (SPI)

## Project Structure

```
campusphere-spi/
├── data/
│   ├── raw/                  # Original datasets (do not modify)
│   │   └── student-mat.csv   # UCI Student Performance - Mathematics
│   └── processed/            # Cleaned, labeled, encoded datasets
├── notebooks/                # Jupyter notebooks for EDA and experiments
├── models/                   # Saved trained models (.joblib files)
├── api/                      # FastAPI service
├── reports/
│   └── figures/              # Charts, confusion matrix, ROC curve
└── README.md

## Dataset
- Source: UCI Student Performance Dataset (Math)
- Variables: 33 columns (grades G1, G2, G3; absences; demographics)
- Target: G3 converted to risk label
  - 0-9  → at_risk
  - 10-12 → needs_review
  - 13-20 → on_track

## Risk Label Logic
G3 < 10          → "at_risk"
10 <= G3 <= 12   → "needs_review"
G3 >= 13         → "on_track"

## Tech Stack
- Python 3.12
- pandas 3.0, numpy 2.4
- scikit-learn 1.8
- matplotlib 3.10, seaborn 0.13
- joblib 1.5
- FastAPI 0.159
