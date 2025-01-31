# Metaboic Syndrome prediction using machine learning Algorithms Models

## Table of Content
---
- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Recommendations](#recommendations)

 ### Project Overview
 ---
 
Our aim was to investigate the correlation between Metabolic Syndrome (MetS) and the use of dolutegravir (DTG) in HIV patients, utilizing data from the VISEND clinical study over 144 weeks. This will include analyzing the efficacy of different DTG-containing regimens like TDF or TAF/XTC/DTG. 

### Data Source
---
VISEND Data: The primary dataset used for this analysis is "VISEND_week_48.csv" file, this file contains 48 weeks follow up information on antropometric measurements, lipid profile, HIV viral and CD4 count test.

### Tools
---
- Excel -data cleaning and capturing [Download here)(https://www.microsoft.com)
- RedCap -data storage [Download here](https://redcap.moh.gov.zm)
- Python -data cleaning and analysis [dowload here](https://www.python.org/downloads)

### Data cleaning and Preparation
---
In the initial data preparation phase, we performed the following tasks:

1. Data loading and inspection.
2. Handling missing values.
3. Data cleaning and formatting.

### Exploring data Analysis
---
EDA involved exploring the VISEND data to answer key questions, such as:

- What is the overall MetS prevalence?
- Which drug regimens show MetS?
- Do all the DTG based regimens show the same prevalence?

### Data Analysis
---
Include some interesting code/features worked with

```python

```

### Results/finding
---
Based on the evaluation metrics and plots, we can draw the following conclusions:

1. **Decision Tree**:
    - Achieved the highest accuracy, precision, recall, and AUC scores on both the training and test datasets.
    - This model appears to be the best performer among all the models evaluated.

2. **Random Forest**:
    - Performed well with high accuracy, precision, recall, and AUC scores.
    - Slightly lower performance compared to the Random Forest model but still a strong contender.

4. **Support Vector Machine (SVM)**:
    - Achieved good accuracy, precision, recall, and AUC scores.
    - Slightly lower performance compared to the top models but still a viable option.
5. **Logistic Regression (L1 Regulaization)**: 
    - Showed competitive performance with high accuracy, precision, recall, and AUC scores.
    - This model is also a strong performer and can be considered for deployment.
6. **Logistic Regression (L2 Regulaization)**:
    -Performed well on the training dataset but showed signs of overfitting with lower test scores.
    - May require further tuning or pruning to improve generalization.

7. **Gaussian Naive Bayes**:
    - Achieved reasonable performance but lower than the top models.
    - Can be considered for simpler and faster predictions.

8. **K-Nearest Neighbors (KNN)**:
    - Showed good performance but slightly lower than the L2 regularized version.
    - Still a viable option for deployment.
9. **XGBoost**:
    - Achieved the lowest performance among all the models.
    - May require further tuning or may not be suitable for this particular dataset.
### Recommendations
---
Based on the evaluation, the **Decision Tree** model is recommended for deployment due to its superior performance across all evaluation metrics. The **Random Forest** and **Support Vector Machine** models are also strong contenders and can be considered as alternative options.
- Encourage life styels changes in food consuption
- Regular exercise for patients on DTG based regimens
- include lipid profile for routine testing 

### Limitation
---
- some patients included at naseline had MetS
- data missing at baseline

### References
1. HIV consolidated manual

**bold**


*Italic*



Analysis steps for predicting Machine Learning in people Living with HIV on a DTG based Regimen
