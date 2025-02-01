# Metaboic Syndrome prediction using machine learning Algorithms Models in People living with HIV on DTG based Regimen

## Table of Content
---
- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Tools](#tools)
- [Data cleaning and Preparation](#data-cleaning-and-preparation)
- [Data Analysis](#data-analysis)
- [Results](#results)
- [Recommendations](#recommendations)
- [Limitation](#limitation)
- [References](#references)
  
 ### Project Overview
 ---
Our aim was to systematically investigate the performance of the seven popular machine learning (ML) algorithms in predicting metabolic syndrome risks in PLHIV on a dolutegravair based regimen. The final analysis included 1034 VISEND participants without MetS at baseline who completed follow-up to 144 weeks. The follow up visists were at weeks 24, 48, 64, 96, 112 and 144. At the follow up visits, anthropometric measurements were done, blood for lipid profile, CD$ count, blood glucose and HIV viral load (HIVVL) were collected and testing. The dataset was divided randomly into two groups, one was the training set (80%) and the other test set (20%). To predict MetS outcomes, the dataset was randomly divided into the training set extreme gradient boosting (XGB), Gaussian naive Bayes (NB), k-nearest neighbors (KNN), L1-penalized logistic regression (LR), L2-penalized logistic regression (LR2), support vector machine with radial basis function (SVM), decision tree (DT), and random forest (RF). Ten-fold cross-validation was employed when developing the model and fine-tuning the hyperparameters in the training set. In the test set, the model's performance was assessed in terms of its clinical significance, discrimination, and calibration. The performances of the predictive models were evaluated using accuracy, sensitivity, specificity, precision, f1-score, and AUC.

### Data Source
---
VISEND clinical trial Data: The primary dataset used for this analysis is "MetSyn_dataset_2025.xlsx" file, this file contains baseline, week-_24, week-_48, week-_64, week-_96, week-_112 and week-_144 follow up visits with information on antropometric measurements, lipid profile, HIVVL and CD4 count test.

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

### Data Analysis
In the initial data preparation phase, we performed the following tasks:

1. **Exploring data Analysis**
    - create functions to display histogram, box plot, count plot and scatter plot for numerical variables
    - Display histogram, box plot, count plot and scatter plot

2. **Explanatory data Analysis**
   - create function to display scatter plot  
   - create function to display correlation heatmap
   - display scatter plots and correction heatmap

3. **Preprocessing/Processing for Machine Learning**
   1. Evaluate if target is even distributed
   2.	Validation spit
   3.	Instatiate column selectors
   4.	Instantiate transformers
   5.	Instantiate pipelines
   6.	Instantiate ColumnTransformer
   7.	Fit and trnasform data
   8.	Insect the results
   9.	Model processors
   10.	Create model pipeline
   11.	Calculate the scores
     -	Accuracy scores
     -	Recall scores
     -	Precision scores
     -	AUC scores
---
Include some interesting code/features worked with

```python
# Install missing packages
%pip install pandas numpy matplotlib seaborn missingno scikit-learn

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import StrMethodFormatter
import matplotlib.gridspec as gridspec
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
# Model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# Dummy Classifier
from sklearn.dummy import DummyClassifier
# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
# Support Vector Classifier
from sklearn.svm import SVC
# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
# KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
# Metrics
from sklearn.metrics import *
#from sklearn import set_config
#set_config(display='diagram')
pd.set_option('display.max.columns', None)
print ('set up complete')

### Loading dataset
# Install missing package
%pip install openpyxl

df = pd.read_excel('/Users/mpanjisiwingwa/Library/Mobile Documents/com~apple~CloudDocs/Desktop/VISEND_Dataset/PhD dataset/clean_data_Set/MetSyn_dataset_2025.xlsx')
# Inspect the data

df.head()

### Function to check for Metabolic Syndrome
# Function to check metabolic syndrome
def check_metabolic_syndrome(row):
    # Define the conditions based on the standard criteria
    criteria = 0
    
    # Abdominal Obesity (assuming 'waist' is in cm and checking for men/women separately)
    if row['waist_circumference'] > 102:  # Men > 102 cm, which is ~102 cm
        criteria += 1
    elif row['waist_circumference'] > 88:  # Women > 88 cm, which is ~88 cm
        criteria += 1

    # Elevated Triglycerides (assuming 'triglycerides' is in mg/dL)
    if row['triglycerides'] >= 1.69: # mmol/L 
        criteria += 1

    # Reduced HDL Cholesterol (assuming 'hdl' is in mg/dL)
    if row['cholesterol_hdl'] < 1.0:  # For men, mmol/L
        criteria += 1
    elif row['cholesterol_hdl'] < 1.3:  # For women, mmol/L
        criteria += 1
    # High Blood Pressure (assuming 'systolic' and 'diastolic' are in mmHg)
    if row['bp_systolic'] >= 130 or row['bp_diastolic'] >= 85:
        criteria += 1

    # Elevated Fasting Glucose (assuming 'fasting_glucose' is in mg/dL)
    if row['blood_sugar'] >= 6.1:  # mmol/L
        criteria += 1

    # If 3 or more criteria are met, return 'MetSyn', otherwise 'No MetSyn'
    return 'MetSyn' if criteria >= 3 else 'No MetSyn'
# Apply the check_metabolic_syndrome function to each row
df['metabolic_syndrome'] = df.apply(check_metabolic_syndrome, axis=1)

# Check if Metabolic syndrome column is created
df.head()

## Inspect the Data
### Display the Row and Column Count
# Display the number of rows and columns for the dataframe

df.shape
print(f'There are {df.shape[0]} rows, and {df.shape[1]} columns.')
print(f'The rows represent {df.shape[0]} observations, and the columns represent {df.shape[1]-1} features and 1 target variable.')

## Display Data Types

# Display the column names and datatypes for each column
# Columns with mixed datatypes are identified as an object datatype

df.dtypes

# Display the column names, count of non-null values, and their datatypes

df.info()

# Data cleaning

# Display the column names
df.columns

### convert float into interger

# list of column that should be converted into integers

new_columns = ['CD4_Count', 'bp_systolic', 'bp_diastolic','date_o_ birth ','Lab_number','Reference_Number(s)']

# define a function that converts columns into intergers
def clean_convert_integers(column):
    #convert column to numeric
    column_numeric = pd.to_numeric(column, errors='coerce')
    # fill NA values with 0
    column_filled = column_numeric.fillna(0)
    #convert the data into interger
    return column_filled.astype(int)

# apply the cleaning and conversion to each column

for col in new_columns:
    if col in df.columns:
        df[col] = clean_convert_integers(df[col])
        
        
#display the data type of the dataframe columns after conversion to verify changes
print(df.dtypes)

# rename columns
mets.rename(columns={'Gender':'Sex','bp_diastolic':'Bp_Diastolic (mmHg)',
                    'bp_systolic':'Bp_Systolic (mmHg)', 'blood_sugar':'Blood_Sugar (mmol/L)', 
                     'triglycerides':'Triglycerides (mmol/L)', 'cholesterol_hdl':'Cholesterol_HDL_(mmol/L)',
                     'waist_circumference':'Waist_Circumference(cm)', 'Age':'Age (yrs)',
                    'bmi':'BMI(kg/m2)', 'Viral_Load ':'Viral_Load(cp/ml)', 'metabolic_syndrome':'Metabolic_Syndrome',
                    'bp_diastolic (mmHg)':'Bp_Diastolic (mmHg)', 'CD4_Count':'CD4_count(cells/µl)',
                    'tobbacco_use':'Tobbacco_Use','Hip_Circumference ':'Hip_Circumference(cm)'}, inplace=True)

# Display the new column names
df.columns

### Remove Unnecessary Columns
# Check if columns exist before dropping them

columns_to_drop = ['Lab_number', 'Reference_Number(s)', 'date_o_ birth ']
existing_columns_to_drop = [col for col in columns_to_drop if col in mets.columns]

# Drop the columns that exist
if existing_columns_to_drop:
	mets.drop(columns=existing_columns_to_drop, inplace=True)
else:
	print("No columns to drop.")

# Display the new column names
df.columns

# express BMI into two decimal places   
mets['BMI(kg/m2)'] = mets['BMI(kg/m2)'].round(2)

### Remove baseline trial number with MetS with their respective subsequent visit
# Filter the dataframe for only the baseline event
baseline_df = mets[mets['Event_Name'] == 'Baseline']

# Identify trial numbers with metabolic syndrome at baseline
if 'Trial_number' in baseline_df.columns:
	trial_numbers_with_metsyn = baseline_df[baseline_df['Metabolic_Syndrome'] == 'MetSyn']['Trial_number']

	# Drop the identified baseline trial numbers with their subsequent visits
	mets = mets[~mets['Trial_number'].isin(trial_numbers_with_metsyn)]
else:
	print("Column 'Trial_number' does not exist in the dataframe.")

# Display the updated dataframe
print(mets.shape)

# Save the updated DataFrame back to csv 
df.to_csv('MetSyn_dataset_2025_cleaned.csv', index=False)

# copy the dataframe into new dataframe
mets = df.copy()

# Dispaly data types
mets.dtypes

# Display the descriptive statistics for the numeric columns
mets.describe().T.round(2)

# Display the descriptive statistics for the non-numeric columns
mets.describe(exclude="number")

# Calculate the count of each category in the 'Metabolic_Syndrome' column
mets_dist = mets['Metabolic_Syndrome'].value_counts().reset_index()
mets_dist.columns = ['Metabolic_Syndrome', 'count']

# Calculate the percentage of each category
mets_dist['percentage'] = (mets_dist['count'] / mets_dist['count'].sum() * 100).round(2)

# Combine the count and percentage into a single string
mets_dist['summary'] = mets_dist['count'].astype(str) + " (" + mets_dist['percentage'].astype(str) + "%)"

# Display the result
print(mets_dist['summary'])

# Calculate the count of unique values and the cardinality for a specific column
column_name = 'Sex'  # Replace with the actual column name
unique_values = mets[column_name].nunique()
cardinality = (mets[column_name].value_counts().sum() / mets.shape[0] * 100).round(2)
# Display the count of unique values and the cardinality for this column
print(f'This column has {unique_values} unique values which is {cardinality}% cardinality.')

# Display the twenty columns
mets.sample(30)

### Remove Unnecessary Rows
# Display the number of duplicate rows in the dataset
print(f'There are {mets.duplicated().sum()} duplicate rows.')
# Remove duplicated rows
mets.drop_duplicates(inplace=True)

# Display the number of rows after removing duplicates
print(f'The number of rows after removing duplicates is {mets.shape[0]}.')

# Missing Values
msno.matrix(mets, figsize=(16,3), labels=True, 
            fontsize=12, sort="descending", color=(0,0,0));
# Plot the missingness
plt.title('Missing Value Status',fontweight='bold')
ax = sns.heatmap(mets.isna().sum().to_frame(),annot=True,fmt='d',cmap='Blues')
ax.set_xlabel('Amount Missing')
plt.show()

# Calculate the count of missing values and their percentage for each column
missing_values_count = mets.isna().sum()
missing_values_percentage = (mets.isna().sum() / mets.shape[0] * 100).round(2)

# Combine the count and percentage into a single string
missing_summary = missing_values_count.astype(str) + " (" + missing_values_percentage.astype(str) + "%)"

# Display the result# Create a function to display supplemental statistics
def column_statistics(column_name, max_unique_values_to_disply=20):
    # Display the count of missing values for this column
    print(f'Missing Values: {mets[column_name].isna().sum()} ({round((mets[column_name].isna().sum())/(mets.shape[0])*100,1)})%')

    # Determine Outliers - Only if this is a numeric column
    if (mets[column_name].dtype == 'int64') | (mets[column_name].dtype == 'float64'):
        # Create outlier filters
        q1 = mets[column_name].quantile(0.25) # 25th percentile
        q3 = mets[column_name].quantile(0.75) # 75th percentile
        iqr = q3 - q1 # Interquartile range
        low_limit = q1 - (1.5 * iqr) # low limit
        high_limit = q3 + (1.5 * iqr) # high limit
        # Create outlier dataframes
        low_mets = mets[(mets[column_name] < low_limit)]
        high_mets = mets[(mets[column_name] > high_limit)]
        # Calculate the outlier counts and percentages
        low_oulier_count = low_mets.shape[0]
        low_outlier_percentge = round(((low_oulier_count)/(mets.shape[0])*100),1)
        high_oulier_count = high_mets.shape[0]
        high_outlier_percentge = round(((high_oulier_count)/(mets.shape[0])*100),1)
        # Display the outlier counts.
        print(f'Outliers: {low_oulier_count} ({low_outlier_percentge})% low, {high_oulier_count} ({high_outlier_percentge})% high')
        
    # Display the count of unique values for this column
    print(f'Unique values: {mets[column_name].nunique()}')

    # Display the unique values including Nan and their counts for this column,
    # if the number of unique values is below the function parameter
    if mets[column_name].nunique() < max_unique_values_to_disply:
        print(mets[column_name].value_counts(dropna=False))                                                 
print(missing_summary)

# Create a function to display supplemental statistics
def column_statistics(column_name, max_unique_values_to_disply=20):
    # Display the count of missing values for this column
    print(f'Missing Values: {mets[column_name].isna().sum()} ({round((mets[column_name].isna().sum())/(mets.shape[0])*100,1)})%')

    # Determine Outliers - Only if this is a numeric column
    if (mets[column_name].dtype == 'int64') | (mets[column_name].dtype == 'float64'):
        # Create outlier filters
        q1 = mets[column_name].quantile(0.25) # 25th percentile
        q3 = mets[column_name].quantile(0.75) # 75th percentile
        iqr = q3 - q1 # Interquartile range
        low_limit = q1 - (1.5 * iqr) # low limit
        high_limit = q3 + (1.5 * iqr) # high limit
        # Create outlier dataframes
        low_mets = mets[(mets[column_name] < low_limit)]
        high_mets = mets[(mets[column_name] > high_limit)]
        # Calculate the outlier counts and percentages
        low_oulier_count = low_mets.shape[0]
        low_outlier_percentge = round(((low_oulier_count)/(mets.shape[0])*100),1)
        high_oulier_count = high_mets.shape[0]
        high_outlier_percentge = round(((high_oulier_count)/(mets.shape[0])*100),1)
        # Display the outlier counts.
        print(f'Outliers: {low_oulier_count} ({low_outlier_percentge})% low, {high_oulier_count} ({high_outlier_percentge})% high')
        
    # Display the count of unique values for this column
    print(f'Unique values: {mets[column_name].nunique()}')

    # Display the unique values including Nan and their counts for this column,
    # if the number of unique values is below the function parameter
    if mets[column_name].nunique() < max_unique_values_to_disply:
        print(mets[column_name].value_counts(dropna=False))                                                 

# Display column statistics for level of education
column_statistics('Educational_Level', 10)

# Fill NaN values with the constant value 'Unknown'
mets.Educational_Level.fillna('Unknown',inplace=True)




```

### Results
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
