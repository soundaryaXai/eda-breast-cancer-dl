# EDA, Visualization & Deep Learning – Breast Cancer Wisconsin (Diagnostic)

**Course:** U21ADP05 – Exploratory Data Analysis and Visualization | **Assignment II (Case Study)**
**Student:** <YOUR NAME> (<ROLL NO>) | AI & DS, KPR Institute of Engineering and Technology

## Problem
Classify breast tumours as **benign or malignant** from 30 numeric cell-nucleus measurements (FNA images) and understand which features drive the diagnosis.

## Dataset
UCI *Breast Cancer Wisconsin (Diagnostic)* – 569 records, 30 features, 1 target.
Dataset link: https://archive.ics.uci.edu/dataset/17 . A copy is included in `data/`; the code loads it directly with `sklearn.datasets.load_breast_cancer`.

## What is done
1. **EDA & preprocessing** – missing values, duplicates, IQR outliers, winsorising, standardisation, stratified 70/15/15 split
2. **Visualisation (8 static + 1 interactive)** – class balance, histograms/KDE, boxplots, correlation heatmap, scatter, pairplot, Normal/Log-normal fit with Q-Q plots, Binomial model
3. **Deep learning** – MLP with grid-search tuning, permutation feature importance
4. **Evaluation** – loss/accuracy vs epoch, confusion matrix, ROC-AUC

## Results (test set, 86 patients)
Accuracy 96.5% | Precision 98.1% | Recall 96.3% | F1 97.2% | ROC-AUC 0.990

## Repository structure
```
eda-breast-cancer-dl/
├── README.md                         # project explanation (this file)
├── requirements.txt                  # Python libraries needed
├── eda_dl_project.py                 # source code (script)
├── EDA_DL_Project.ipynb              # source code (executed notebook)
├── data/
│   └── breast_cancer_wisconsin.csv   # dataset copy (569 rows x 31 columns)
├── report/
│   └── EDA_Case_Study_Report.pdf     # project report
├── figures/                          # 13 charts (8 EDA + 5 model-evaluation PNGs)
└── outputs/
    ├── summary_statistics.csv
    ├── hyperparameter_tuning.csv
    ├── results.json
    └── interactive_scatter.html      # interactive Plotly chart
```

## How to run
```bash
pip install -r requirements.txt
python eda_dl_project.py          # creates figures/ and outputs/
# or: jupyter notebook EDA_DL_Project.ipynb
```
