# Machine Learning-Based Prediction of Metabolic Syndrome in HIV Cohorts Receiving Dolutegravir-Based ART

![R Version](https://img.shields.io/badge/R-4.5.1-blue?logo=r)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Pipeline-success?logo=r)
![Clinical Research](https://img.shields.io/badge/Clinical-HIV%20Research-red?logo=medrxiv)
![Project Status](https://img.shields.io/badge/Status-Active_Research-orange?logo=github)
![License: MIT](https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative)
[![GitHub Profile](https://img.shields.io/badge/GitHub-MpanjiSiwingwa-black?logo=github)](https://github.com/MpanjiSiwingwa)

Predictive machine learning pipeline for identifying incident metabolic syndrome (MetS) among people living with HIV (PLHIV) receiving dolutegravir-based antiretroviral therapy (ART) using longitudinal clinical trial data from Zambia.

---

## 📖 Table of Contents

- [Highlights](#-highlights)
- [Project Overview](#-project-overview)
- [Study Workflow](#-study-workflow)
- [Dataset](#-dataset)
- [Machine Learning Models](#-machine-learning-models)
- [Feature Selection & Explainability](#-feature-selection--explainability)
- [Model Evaluation](#-model-evaluation)
- [Repository Structure](#-repository-structure)
- [Getting Started](#️-getting-started)
- [Reproducible Environment](#-reproducible-environment)
- [System Requirements](#-system-requirements)
- [Approximate Runtime](#️-approximate-runtime)
- [Figures](#-figures)
- [Results](#-results)
- [Key Findings](#-key-findings)
- [Manuscript–Code Correspondence](#-manuscriptcode-correspondence)
- [Data Availability](#-data-availability)
- [Ethical Approval](#-ethical-approval)
- [Reporting Standards](#-reporting-standards)
- [Strengths](#-strengths)
- [Limitations](#️-limitations)
- [Future Work](#-future-work)
- [References](#-references)
- [Citation](#-citation)
- [Repository Citation](#-repository-citation)
- [License](#-license)
- [Contact](#-contact)
- [Status](#-status)

---

## 🔬 Highlights

- Developed and compared **9 supervised machine learning models**
- Applied **SHAP explainability** for interpretable AI
- Performed **calibration and decision curve analysis**
- Built using a **fully reproducible R workflow**
- Based on a **longitudinal HIV cohort from Zambia**
- Focused on **clinically interpretable prediction models**

---

## 📘 Project Overview

Metabolic syndrome (MetS) is increasingly prevalent among people living with HIV, particularly among individuals receiving dolutegravir-based antiretroviral therapy. Early identification of high-risk individuals may support preventive interventions and improve long-term cardiovascular outcomes.

This repository contains the complete machine learning workflow used to develop predictive models for metabolic syndrome using longitudinal HIV clinical data from the VISEND study conducted in Zambia.

The project integrates:
- Clinical epidemiology
- Machine learning
- Explainable artificial intelligence (XAI)
- Reproducible research practices

---

## 🖼️ Study Workflow

![Workflow](figures/Figure%201_Framework%20for%20Predicting%20MetSyn_070526.png)

---

## 📊 Dataset

- **Population:** Adults living with HIV receiving dolutegravir-based ART
- **Follow-up:** Up to 144 weeks
- **Source:** VISEND study, University Teaching Hospital, Lusaka, Zambia
- **Outcome:** Incident metabolic syndrome (MetS)
- **Predictors:** Demographic, clinical, HIV-related, lifestyle, and laboratory variables

### Leakage Prevention

Diagnostic variables directly used in the harmonized definition of metabolic syndrome (waist circumference, hip Circumference, triglycerides, HDL cholesterol, blood pressure, and fasting glucose) were excluded from predictive modeling to reduce target leakage and improve clinical validity.

---

## 🤖 Machine Learning Models

| Model | Description |
|---|---|
| Logistic Regression | Regularized generalized linear model |
| Random Forest | Ensemble tree-based classifier |
| XGBoost | Gradient boosting framework |
| LightGBM | Efficient gradient boosting |
| SVM (Linear) | Linear kernel support vector machine |
| SVM (Radial) | Non-linear kernel support vector machine |
| Decision Tree | Recursive partitioning model |
| K-Nearest Neighbours | Distance-based classifier |
| Naïve Bayes | Probabilistic classifier |

---

## 🧩 Feature Selection & Explainability

### Feature Selection
- Boruta algorithm
- Random Forest variable importance
- Variance Inflation Factor (VIF) diagnostics

### Explainability
- SHAP values for:
  - feature importance
  - local interpretability
  - clinical insight generation

---

## 📈 Model Evaluation

### Validation Strategy
- Stratified train/test split
- Repeated 10-fold cross-validation
- Hyperparameter tuning

### Performance Metrics
- ROC-AUC
- PR-AUC
- Accuracy
- Sensitivity
- Specificity
- Matthews Correlation Coefficient (MCC)
- Youden Index
- Brier Score

### Additional Analyses
- Calibration analysis
- Decision curve analysis
- Generalized additive model (GAM) diagnostics

---

## 📂 Repository Structure

```text
MetSyn-ML-Prediction/
│
├── README.md
├── LICENSE
├── renv.lock
├── .gitignore
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
├── results/
├── manuscript/
└── sessionInfo.txt
```

---

## 🛠️ Getting Started

### 🚀 Clone the Repository

```bash
git clone https://github.com/MpanjiSiwingwa/Metabolic-syndrome-prediction-using-machine-learning.git
cd Metabolic-syndrome-prediction-using-machine-learning
```

---

### ⚙️ Install Dependencies

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
  "mice",
  "renv"
))
```

---

### ▶️ Run the Pipeline

```r
source("scripts/01_data_cleaning.R")
source("scripts/02_feature_selection.R")
source("scripts/03_model_training.R")
source("scripts/04_model_evaluation.R")
source("scripts/05_visualization.R")
```

Alternatively:

```bash
Rscript scripts/01_data_cleaning.R
```

---

## 📦 Reproducible Environment

Package versions were managed using:
- R 4.5.1
- `renv`
- `sessionInfo()`

Restore the computational environment using:

```r
renv::restore()
```

Export session information:

```r
writeLines(capture.output(sessionInfo()), "sessionInfo.txt")
```

---

## 💻 System Requirements

- R ≥ 4.5.1
- macOS, Linux, or Windows
- Recommended RAM: ≥16 GB
- Multi-core CPU recommended for model training

---

## ⏱️ Approximate Runtime

| Step | Estimated Runtime |
|---|---|
| Data cleaning | 2–5 min |
| Feature selection | 5–15 min |
| Model training | 20–60 min |
| SHAP analysis | 10–30 min |

---

## 📊 Figures

Key manuscript figures are available in the `figures/` directory.

### Main Figures
- Figure 1. Predictive modeling workflow
- Figure 2. Calibration analysis
- Figure 3. ROC curve comparison
- Figure 4. Decision curve analysis
- Figure 5. SHAP feature importance
- Figure 6. Sensitivity analysis

### Supplementary Figures
- Random Forest feature importance
- Boruta feature selection
- GAM effect plots

---

## 📈 Results

The performance of the primary machine learning models is summarized below.

| Model | ROC-AUC (95% CI) | PR-AUC (%) | Accuracy (%) |
|---|---|---|---|
| Decision Tree | 53.4 (45.1–61.8) | 22.2 | 44.7 |
| K-NN | 63.7 (55.1–72.2) | 29.9 | 69.0 |
| LightGBM | 60.3 (51.9–68.7) | 26.6 | 51.8 |
| Logistic Regression | 63.3 (54.8–71.8) | 32.1 | 70.2 |
| Naïve Bayes | 62.0 (53.6–70.4) | 28.2 | 61.6 |
| XGBoost | 59.6 (51.1–68.1) | 30.7 | 45.5 |
| Random Forest | 61.6 (52.6–70.5) | 28.5 | 70.2 |
| SVM Linear | 49.3 (39.8–58.9) | 22.0 | 71.0 |
| **SVM Radial** | **64.2 (56.1–72.3)** | **31.8** | **52.2** |

The best discrimination performance was observed for the radial SVM model, followed closely by logistic regression. Despite modest discrimination, calibration and decision curve analyses suggested potential clinical utility for identifying higher-risk individuals.

---

## 🔑 Key Findings

- Radial SVM and logistic regression showed the best discrimination performance
- XGBoost demonstrated higher sensitivity with lower specificity
- Calibration improved after isotonic regression
- SHAP identified age, sex, viral load, and alcohol use as key predictors
- GAM analysis revealed non-linear CD4 effects
- Complex machine learning models provided only modest gains over logistic regression

---

## 🧪 Manuscript–Code Correspondence

| Manuscript Component | Repository Script |
|---|---|
| Data preprocessing | `01_data_cleaning.R` |
| Feature selection | `02_feature_selection.R` |
| Model development | `03_model_training.R` |
| Model evaluation | `04_model_evaluation.R` |
| Visualization and SHAP | `05_visualization.R` |

---

## 🔐 Data Availability

The VISEND clinical dataset used in this study contains sensitive participant information and is not publicly available due to ethical and institutional restrictions.

Researchers interested in accessing de-identified data for scientific collaboration may contact the corresponding author subject to institutional approvals and data-sharing agreements.

All scripts required to reproduce the analyses are fully available in this repository.

---

## 🧾 Ethical Approval

The VISEND study received ethical approval from the University of Zambia Biomedical Research Ethics Committee (UNZABREC). All participants provided informed consent prior to enrollment.

---

## 📑 Reporting Standards

This repository and accompanying manuscript were developed in alignment with:
- TRIPOD reporting recommendations
- TRIPOD-AI guidance principles
- Transparent and reproducible machine learning practices in clinical research

---

## ✅ Strengths

- Focused on PLHIV receiving contemporary ART
- Comprehensive evaluation of multiple machine learning algorithms
- Included calibration and clinical utility assessment
- Emphasized explainability and reproducibility

---

## ⚠️ Limitations

- External and prospective validation remain necessary before clinical implementation
- Important predictors such as diet, physical activity, and genomics were unavailable
- Generalisability across populations and treatment regimens may be limited
- Lack of external datasets restricted external validation opportunities

This study represents an internally validated predictive modeling framework. External validation in geographically and clinically independent cohorts remains necessary prior to clinical implementation.

---

## 🔮 Future Work

- External validation in independent cohorts
- Integration of genomic predictors
- Temporal prediction modeling
- Clinical risk score development
- EMR/CDSS integration

---

## 📚 References

1. Hamooya BM, Mulenga LB, Masenga SK, et al. *Medicine (Baltimore).* 2021;100(14).  
2. Zambia Ministry of Health. Zambia Consolidated HIV Guidelines.  
3. Saklayen MG. *Curr Hypertens Rep.* 2018;20(2):12.  
4. Moons KGM, Altman DG, Reitsma JB, et al. *Ann Intern Med.* 2015;162:W1–73.  
5. Vickers AJ, Elkin EB. *Med Decis Making.* 2006;26(6):565–574.

---

## 📌 Citation

Siwingwa M, et al. *Machine Learning-Based Prediction of Metabolic Syndrome in HIV Cohorts Receiving Dolutegravir-Based ART.* Manuscript in preparation, 2026.

---

## 📖 Repository Citation

Siwingwa M. *Metabolic Syndrome Prediction Using Machine Learning in HIV Cohorts Receiving Dolutegravir-Based ART* [GitHub repository]. 2026.

Available at:  
https://github.com/MpanjiSiwingwa/Metabolic-syndrome-prediction-using-machine-learning

---

## 📜 License

This project is licensed under the MIT License.

---

## 📬 Contact

**Mpanji Siwingwa**  
PhD Researcher | Machine Learning | HIV Research | Bioinformatics

- GitHub: https://github.com/MpanjiSiwingwa
- Email: mpanjisiwingwa@gmail.com
- LinkedIn: https://linkedin.com/in/mpanji-siwingwa-b0272a74
- ORCID: https://orcid.org/0000-0002-3623-2108

---

## 📌 Status

This repository accompanies an ongoing research project and will continue evolving as additional analyses and validation studies are completed.
