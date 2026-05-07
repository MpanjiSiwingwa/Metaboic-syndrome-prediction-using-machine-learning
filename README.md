# Machine Learning‑Based Prediction of Metabolic Syndrome in HIV Cohorts Receiving Dolutegravir‑Based ART

![R Version](https://img.shields.io/badge/R-4.5.1-blue?logo=r)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Pipeline-success?logo=r)
![Clinical Research](https://img.shields.io/badge/Clinical-HIV%20Research-red?logo=medrxiv)
![Project Status](https://img.shields.io/badge/Status-Active_Research-orange?logo=github)
![License: MIT](https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative)
[![GitHub Profile](https://img.shields.io/badge/GitHub-MpanjiSiwingwa-black?logo=github)](https://github.com/MpanjiSiwingwa)

Predictive machine learning pipeline for identifying incident metabolic syndrome (MetS) among people living with HIV (PLHIV) receiving dolutegravir-based antiretroviral therapy (ART) using longitudinal clinical trial data from Zambia.

---
## Table of Contents

- [Highlights](#highlights)
- [Project Overview](#project-overview)
- [Workflow](#workflow)
- [Dataset](#-dataset)
- [Machine Learning Models](#-machine-learning-models)
- [Feature Selection & Explainability](#-feature-selection--explainability)
- [Model Evaluation](#-model-evaluation)
- [Key Findings](#-key-findings)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Installation](#️-installation)
- [Running the Pipeline](#️-running-the-pipeline)
- [Figures](#-figures)
- [Reproducibility](#-reproducibility)
- [Strengths](#-strengths)
- [Limitations](#️-limitations)
- [Future Work](#-future-work)
- [References](#-references)
- [Citation](#-citation)
- [License](#-license)
- [Contact](#-contact)
- [Status](#-status)
---
# Highlights

- Developed and compared **9 supervised machine learning models**
- Applied **SHAP explainability** for interpretable AI
- Performed **calibration and decision curve analysis**
- Built using a **fully reproducible R workflow**
- Based on a **longitudinal HIV cohort from Zambia**
- Focused on **clinically interpretable prediction models**

---

# Project Overview

Metabolic syndrome (MetS) is increasingly prevalent among people living with HIV, particularly among individuals receiving dolutegravir-based antiretroviral therapy. Early identification of high-risk individuals may support preventive interventions and improve long-term cardiovascular outcomes.

This repository contains the complete machine learning workflow used to develop predictive models for metabolic syndrome using longitudinal HIV clinical data from the VISEND study conducted in Zambia.

The project combines:
- clinical epidemiology,
- machine learning,
- explainable AI,
- and reproducible research practices.

---

# Workflow

```text
Raw Clinical Data
        ↓
Data Cleaning & Quality Control
        ↓
Feature Engineering
        ↓
Feature Selection
(Boruta + Random Forest + VIF)
        ↓
Train/Test Split
        ↓
Machine Learning Model Training
        ↓
Model Evaluation
        ↓
Calibration Analysis
        ↓
Explainability (SHAP)
        ↓
Clinical Utility Assessment
```
---
### 📊 Dataset
- Population: Adults living with HIV receiving dolutegravir‑based ART
- Follow‑up: Up to 144 weeks
- Source: VISEND study, University Teaching Hospital, Lusaka, Zambia
- Outcome: Incident metabolic syndrome (MetS)
- Predictors: demographics, clinical, HIV‑related, lifestyle, laboratory variables
- Leakage prevention: diagnostic MetS variables excluded (waist, BP, HDL, triglycerides, glucose)
---
### 🤖 Machine Learning Models

| Model             | Description                  |
| ----------------- | ---------------------------- |
| Logistic Regression | Regularized GLM             |
| Random Forest       | Ensemble tree‑based         |
| XGBoost             | Gradient boosting           |
| LightGBM            | Efficient boosting          |
| SVM (Linear)        | Linear kernel               |
| SVM (Radial)        | Non‑linear kernel           |
| Decision Tree       | Recursive partitioning      |
| KNN                 | k‑nearest neighbours        |
| Naïve Bayes         | Probabilistic classifier    |

---
### **🧩 Feature Selection & Explainability**
- **Selection:** Boruta, Random Forest importance, VIF diagnostics
- **Explainability:** SHAP values for feature importance, interpretability, and clinical insight
---
### 📈 Model Evaluation
- **Validation:** stratified train/test split, repeated 10‑fold CV, hyperparameter tuning
- **Metrics:** ROC‑AUC, PR‑AUC, accuracy, sensitivity, specificity, MCC, Youden index, Brier score
- **Additional:** calibration analysis, decision curve analysis, GAM diagnostics
---
### 🔑 Key Findings
- Radial SVM & logistic regression best (AUC ≈0.64)
- XGBoost high sensitivity, lower specificity
- Calibration improved with isotonic regression
- SHAP confirmed age, sex, viral load, alcohol use as key predictors
- GAM revealed non‑linear CD4 effects
- Complex models offered modest gains over logistic regression
---
### 📂 Repository Structure
```text
MetSyn-ML-Prediction/
│
├── README.md
├── LICENSE
├── data/
│   └── README_data.md
├── scripts/
│   ├── 01_data_cleaning.R
│   ├── 02_feature_selection.R
│   ├── 03_model_training.R
│   ├── 04_model_evaluation.R
│   ├── 05_visualization.R
│   └── utils.R
├── figures/
│   ├── ROC_Curves.png
│   ├── PR_Curves.png
│   ├── SHAP_Importance.png
│   └── Workflow_Diagram.png
├── results/
│   ├── model_metrics.csv
│   ├── calibration_results.csv
│   └── shap_outputs.csv
└── manuscript/
    └── supplementary_materials.pdf
```
---
### 🚀 Quick Start
```bash
git clone https://github.com/MpanjiSiwingwa/Metabolic-syndrome-prediction-using-machine-learning.git
cd Metabolic-syndrome-prediction-using-machine-learning
Rscript scripts/01_data_cleaning.R
```
---
### ⚙️ Installation

```r
install.packages(c(
  "tidyverse",
  "caret",
  "recipes",
  "glmnet",
  "randomForest",
  "xgboost",
  "lightgbm",
  "Boruta",
  "fastshap",
  "pROC",
  "PRROC",
  "rmda",
  "mgcv",
  "car",
  "mice"
))
```
---
### ▶️ Running the Pipeline

```r
source("scripts/01_data_cleaning.R")
source("scripts/02_feature_selection.R")
source("scripts/03_model_training.R")
source("scripts/04_model_evaluation.R")
source("scripts/05_visualization.R")
```
---
### 📊 Figures
- ROC curve comparison
- Precision‑Recall curves
- SHAP feature importance
---
### 🔄 Reproducibility

- R version: 4.5.1
- Random seed: 123
- Stratified train/test split
- Repeated 10-fold CV
- Fully scripted workflow

Export session information:

```r
writeLines(capture.output(sessionInfo()), "sessionInfo.txt")
```
---
### ✅ Strengths
- Focused on PLHIV receiving contemporary ART, an underrepresented population in predictive modeling.
- Systematic evaluation of multiple machine learning algorithms.
- Comprehensive assessment using discrimination, calibration, and decision curve analysis.

### ⚠️ Limitations
- External and prospective validation is required before clinical application.
- Important predictors (diet, physical activity, genetics) were unavailable.
- Generalisability may be limited across different populations and treatment regimens.
- Calibration partly relied on the Hosmer–Lemeshow test, which has known limitations.
- Lack of suitable external datasets restricted external validation.

### 🔮 Future Work
- External validation in independent cohorts
- Genomic predictor integration
- Temporal prediction models
- Clinical risk score development
- EMR/CDSS integration
---
###  📚 References
1. Hamooya BM, Mulenga LB, Masenga SK, Fwemba I, Chirwa L, Siwingwa M, et al. Metabolic syndrome in Zambian adults with human immunodeficiency virus on antiretroviral therapy: Prevalence and associated factors. Medicine (United States). 2021;100(14). doi:10.1097/MD.0000000000025236
2. Zambia Ministry of Health. Zambia Consolidated Guidelines for Treatment and Prevention of HIV Infection. Lusaka, Zambia.
3. Saklayen MG. The Global Epidemic of the Metabolic Syndrome. Curr Hypertens Rep. 2018;20(2):12. doi:10.1007/s11906-018-0812-z
4. Moons KGM, Altman DG, Reitsma JB, Ioannidis J, Macaskill P, Steyerberg EW, et al. Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD): Explanation and Elaboration. Ann Intern Med. 2015;162:W1–73.
5. Vickers AJ, Elkin EB. Decision curve analysis: A novel method for evaluating prediction models. Med Decis Making. 2006;26(6):565–74.doi:10.1177/0272989X06295361
---
### 📌 Citation
Siwingwa M, et al. Machine Learning‑Based Prediction of Metabolic Syndrome in HIV Cohorts Receiving Dolutegravir‑Based ART. (Manuscript in preparation, 2026).

# 📜 License
This project is licensed under the MIT License.


## 📬 Contact
**Mpanji Siwingwa**  
PhD Researcher | Machine Learning | HIV Research | Bioinformatics  

- 🐙 GitHub: [https://github.com/MpanjiSiwingwa](https://github.com/MpanjiSiwingwa)  
- ✉️ Email: [mpanjisiwingwa@gmail.com](mailto:mpanjisiwingwa@gmail.com)  
- 🔗 LinkedIn: [linkedin.com/in/mpanji-siwingwa-b0272a74](https://linkedin.com/in/mpanji-siwingwa-b0272a74)  
- 🔗 ORCID: [orcid.org/0000-0002-3623-2108](https://orcid.org/0000-0002-3623-2108)  

### 📌 Status
This repository accompanies an ongoing research project and will evolve as additional analyses and validation studies are completed.
