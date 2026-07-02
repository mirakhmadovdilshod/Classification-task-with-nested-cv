# Breast Cancer Diagnosis Classification — Nested Cross-Validation

Classification of breast tumor diagnosis (Malignant vs. Benign) using supervised machine learning with a Stratified Nested Cross-Validation scheme.

Course project for **AI for Medicine**

## Overview

This project builds and compares four supervised classification models on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, with a strong emphasis on methodological rigor: preventing data leakage, obtaining an unbiased performance estimate, and ensuring full reproducibility — rather than optimizing for accuracy alone.

## Dataset

- **Source:** Wisconsin Diagnostic Breast Cancer (WDBC), via Kaggle
- **Samples:** 569 patients
- **Features:** 30 numerical measurements of cell nuclei (mean, standard error, and worst value for 10 base characteristics: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
- **Target:** Diagnosis — Malignant (0) or Benign (1)
- **Class balance:** 357 Benign / 212 Malignant

## Methodology

1. **Exploratory Data Analysis** — class distribution, feature distributions, correlation matrix
2. **Stratified train/test split** (80/20), test set held out until final evaluation
3. **Preprocessing** — StandardScaler, fit only on training data (inside a Pipeline)
4. **Data Leakage Checklist** — based on the taxonomy from Kapoor & Narayanan, *Patterns* (2023): L1.1–L1.4, L2, L3.2
5. **Model comparison** — Logistic Regression, Random Forest, SVM, K-Nearest Neighbors, each wrapped in an sklearn Pipeline (Scaler + Classifier)
6. **Stratified Nested Cross-Validation** — outer loop (5-fold) for unbiased evaluation, inner loop (3-fold, via GridSearchCV) for hyperparameter tuning
7. **Final model** — retrained on the entire dataset using the best hyperparameters
8. **Evaluation** — ROC AUC (primary metric, robust to class imbalance), confusion matrix, precision, recall, F1-score
9. **Feature importance** — inspection of Logistic Regression coefficients

## Results

| Model | Nested CV ROC AUC | Test ROC AUC |
|---|---|---|
| **Logistic Regression** (final model) | 0.994 ± 0.010 | 0.995 |
| SVM | 0.995 ± 0.006 | 0.995 |
| Random Forest | 0.988 ± 0.009 | 0.994 |
| KNN | 0.988 ± 0.012 | 0.994 |

The near-identical Nested CV and held-out test scores indicate the model generalizes well, with no evidence of overfitting.

**Best hyperparameters (Logistic Regression):** `C = 1`

**Most important features:** predominantly "worst" cell measurements (worst texture, worst radius, worst area) — consistent with the clinical intuition that malignancy is driven by the most abnormal cells in a sample.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── notebook/
│   └── breast_cancer_classification_nestedCV.ipynb
└── report/
    └── AI_for_Medicine_Project_Report.pdf
```

## How to run

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open `notebook/breast_cancer_classification_nestedCV.ipynb` in Jupyter or Google Colab
4. Run cells sequentially from top to bottom

The dataset is loaded from a CSV file (`data.csv`, WDBC source). Update the file path in the notebook to point to your local copy or Google Drive location.

## Reproducibility

- `RANDOM_STATE = 42` fixed across all splits, cross-validation folds, and models
- Library versions specified in `requirements.txt`
- Full methodology and data leakage prevention steps documented in the notebook and report

## Limitations

- Small dataset (569 samples) — generalization to more diverse populations not verified
- Single data source — no external validation on data from a different hospital/scanner
- Classification threshold fixed at the default 0.5, not tuned for clinical false-negative priorities

## License

This project is for educational purposes as part of a university course. The model is not intended for clinical use without proper validation and regulatory approval.

## Author

Course project by MDC

