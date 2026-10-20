# Diabetes Risk Factors: An Exploratory Data Analysis

A reproducible data-science project built for the CISC7204 course practice:
an exploratory data analysis (EDA) of the classic **Pima Indians Diabetes
dataset**, ending with a baseline logistic-regression classifier.

## 1. Problem / Research Question

Can routine health indicators — plasma glucose, body mass index (BMI), age,
blood pressure, insulin, number of pregnancies, and diabetes pedigree — be
used to distinguish patients who develop diabetes from those who do not?

The project answers two concrete sub-questions:

1. **Which indicators are most strongly associated** with a diabetes outcome?
2. **How well can a simple logistic-regression model** separate diabetic from
   non-diabetic patients using only these routine measurements?

## 2. Dataset Source and Usage Permissions

| Item | Detail |
| --- | --- |
| Dataset | Pima Indians Diabetes Database |
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes |
| Size | 768 records, 8 features + 1 outcome (500 non-diabetic, 268 diabetic) |
| Usage permission | Publicly available dataset, released for research and educational use |
| Note | Contains no personal identifiers; it is a de-identified aggregate sample |

The raw file `data/diabetes_raw.csv` is stored in this repository so the
analysis can be reproduced offline. **No passwords, API keys, personal data,
or restricted datasets are included** in this repository.

## 3. Methods and Tools Used

- **Tools:** Python 3, pandas, NumPy, Matplotlib, Seaborn, scikit-learn
  (see `requirements.txt`).
- **Data cleaning:** values of 0 that are physiologically impossible
  (glucose, blood pressure, skin thickness, insulin, BMI) were treated as
  missing; 652 such values were imputed with the column median.
- **Exploratory analysis:** summary statistics, class balance, correlation
  analysis, and distribution comparisons (4 figures in `figures/`).
- **Modelling:** logistic regression with an 80/20 stratified
  train/test split (random seed 42); evaluated with accuracy, precision,
  recall, and ROC-AUC. Feature influence shown via model coefficients.

## 4. Key Findings and One Limitation

**Key findings (all reproducible by running `analysis.py`):**

1. **Glucose is the strongest single risk indicator** — correlation with the
   outcome is r = 0.49. Mean glucose is 142.1 mg/dL in diabetic patients vs
   110.7 mg/dL in non-diabetic patients (see `fig1_glucose_by_outcome.png`).
2. **BMI (r = 0.31) and age (r = 0.24)** are the next strongest factors;
   glucose, BMI and age are the top three coefficients in the model
   (see `fig5_feature_importance.png`).
3. **Simple baseline model works reasonably:** logistic regression reaches
   **70.1% accuracy** and **ROC-AUC = 0.813** on the held-out test set.
4. Strong feature inter-correlations exist (e.g., pregnancies–age r = 0.54,
   skin thickness–BMI r = 0.54), so individual coefficients should be read
   with caution (see `fig2_correlation_heatmap.png`).

**One limitation:** the dataset is a single, decades-old clinical cohort
(N = 768) of one ethnic group, so the findings may not generalize to other
populations. The baseline model also shows low recall (50.0%) on the
minority class, meaning it misses half of the diabetic patients — a tuned or
more flexible model would be needed before any real-world use.

## 5. How to Run the Analysis

```bash
# 1. Clone / download this repository and enter the folder
git clone https://github.com/[YOUR-GITHUB-USERNAME]/data-science-project.git
cd data-science-project

# 2. (Recommended) create and activate a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the analysis (reads data/diabetes_raw.csv, writes figures/ and
#    data/diabetes_clean.csv, prints summary + model metrics)
python analysis.py
```

Expected outputs:

```
data/diabetes_clean.csv            cleaned dataset (median-imputed)
figures/fig1_glucose_by_outcome.png
figures/fig2_correlation_heatmap.png
figures/fig3_bmi_age_scatter.png
figures/fig4_pregnancies_age_boxplot.png
figures/fig5_feature_importance.png
```

## Repository Structure

```
data-science-project/
├── README.md
├── requirements.txt
├── analysis.py
├── data/
│   └── diabetes_raw.csv          # public UCI dataset (de-identified)
└── figures/                      # 5 visualizations produced by analysis.py
```

---
*Author: [YOUR NAME] — CISC7204 Practice 09: Build Your Data Science Portfolio*
