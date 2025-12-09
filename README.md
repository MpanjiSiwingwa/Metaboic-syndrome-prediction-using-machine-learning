# Machine Learning-based Prediction of Metabolic Syndrome Enhances Risk Stratification in Dolutegravir-Treated HIV Cohorts


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
We analyzed a retrospective cohort of 1027 PLHIV without MetS at baseline, followed-up for 144 weeks. Participants were randomly allocated into training (70%) and test (30%) sets. Logistic regression (LR), support vector machine (SVM-linear/radial), decision tree (DT), random forest (RF), k-nearest neighbors (KNN), Gaussian naive Bayes (NB), lightGBM and XGBoost were used to predict MetS outcomes.  Models were trained using five-fold cross-validation with hyperparameter tuning. Performance was assessed on the test set using discrimination, calibration, and classification metrics. RF was used to rank the importance of the predictors. In this study, we sought to develop and validate a suite of supervised machine learning algorithms to predict the onset of MetS in a Zambian cohort of PLHIV on dolutegravir (DTG)-based regimens.

### Data Source
---
VISEND clinical trial Data: The primary dataset used for this analysis is "MetSyn_dataset_2025.xlsx" file, this file contains baseline and week-_144 follow up visits with information on antropometric measurements, lipid profile, HIVVL and CD4 count test.

### Tools
---
- Excel -data cleaning and capturing [Download here)(https://www.microsoft.com)
- RedCap -data storage [Download here](https://redcap.moh.gov.zm)
- R — data cleaning and analysis [download here](https://cran.r-project.org)

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

# Display the unique values for this column
print(mets['Educational_Level'].unique())

# Display column statistics diabetis mellitus
column_statistics('Diabetes_Mellitus_status', 10)

# Fill NaN values with the constant value 'Unknown'
mets.Diabetes_Mellitus_status.fillna('Unknown',inplace=True)

# Display the unique values for this column
print(mets['Diabetes_Mellitus_status'].unique())

# Display column statistics diabetis mellitus
column_statistics('Tobbacco_Use', 10)

# Fill NaN values with the constant value 'Unknown'
mets.Tobbacco_Use.fillna('Unknown',inplace=True)

# Display the unique values for this column
print(mets['Tobbacco_Use'].unique())

# Display column statistics diabetis mellitus
column_statistics('Alcohol_Consuption', 10)

# Fill NaN values with the constant value 'Unknown'
mets.Alcohol_Consuption .fillna('Unknown',inplace=True)

# Display the unique values for this column
print(mets['Alcohol_Consuption'].unique())

# Display column statistics drug_code
column_statistics('Regimen_Type', 10)

# Display column statistics Event_Name
column_statistics('Event_Name', 10)

## Numeric Columns

# Display column statistics of Age
column_statistics('Age (yrs)', 1)

# Display column statistics of BMI 
column_statistics('BMI(kg/m2)', 1)

# Display column statistics of waist_circumference
column_statistics('Waist_Circumference(cm)', 1)

# Display column statistics of Hip_Circumference(cm)
column_statistics('Hip_Circumference(cm)', 1)

# Display column statistics of CD4_count
column_statistics('CD4_count(cells/µl)', 1)

# Display column statistics of Viral_Load(cp/ml)
column_statistics('Viral_Load(cp/ml)', 1)

# Display column statistics of HDL cholesterol
column_statistics('Cholesterol_HDL_(mmol/L)', 1)

# Display column statistics of Triglycerides (mmol/L)
column_statistics('Triglycerides (mmol/L)', 1)

# Display the column names to verify the correct name
print(mets.columns)

# Display column statistics of Blood_Sugar
column_statistics('Blood_Glucose (mmol/L)', 1)

# fill NA with mean value

mets[['BMI(kg/m2)', 'Waist_Circumference(cm)', 'Hip_Circumference(cm)', 'Viral_Load(cp/ml)', 'Blood_Glucose (mmol/L)', 'Triglycerides (mmol/L)', 'Cholesterol_HDL_(mmol/L)']] = mets[['BMI(kg/m2)', 'Waist_Circumference(cm)', 'Hip_Circumference(cm)', 'Viral_Load(cp/ml)', 'Blood_Glucose (mmol/L)', 'Triglycerides (mmol/L)', 'Cholesterol_HDL_(mmol/L)']].fillna(mets[['BMI(kg/m2)', 'Waist_Circumference(cm)', 'Hip_Circumference(cm)', 'Viral_Load(cp/ml)', 'Triglycerides (mmol/L)', 'Cholesterol_HDL_(mmol/L)']].mean())

# Display the count of missing values by column
print(mets.isna().sum())

mets.info()

# describe data
mets.describe().T.round(2)

# Inspect column Datatypes for Errors
# Display the descriptive statistics for the non-numeric columns
mets.describe(exclude="number")

# display the unique values for sex column 
print(mets['Sex'].unique())

# Calculate value counts and their percentages
gender_counts = mets['Sex'].value_counts()
gender_percentages = mets['Sex'].value_counts(normalize=True) * 100

# Combine counts and percentages into a single string
gender_summary = gender_counts.astype(str) + " (" + gender_percentages.round(2).astype(str) + "%)"

# Display the result
print(gender_summary)

# Filter the dataframe for only the baseline event
baseline_df = mets[mets['Event_Name'] == 'Baseline']

# Calculate value counts and their percentages for the 'Sex' column
baseline_gender_counts = baseline_df['Sex'].value_counts()
baseline_gender_percentages = baseline_df['Sex'].value_counts(normalize=True) * 100

# Combine counts and percentages into a single string
baseline_gender_summary = baseline_gender_counts.astype(str) + " (" + baseline_gender_percentages.round(2).astype(str) + "%)"

# Display the result
print(baseline_gender_summary)

# Regimen Types columns
# Rename the column 'Drug_Code ' to 'Regimen_Type'
mets.rename(columns={'Drug_Code ': 'Regimen_Type'}, inplace=True)

# Display the updated column names to verify the change
print(mets.columns)

# display the unique values for this column 
print(mets['Regimen_Type'].unique())

# Calculate value counts and their percentages
drugcode_counts = mets['Regimen_Type'].value_counts()
drugcode_percentages = mets['Regimen_Type'].value_counts(normalize=True) * 100

# Combine counts and percentages into a single string
drugcode_summary = drugcode_counts.astype(str) + " (" + drugcode_percentages.round(2).astype(str) + "%)"

# Display the result
print(drugcode_summary)

### Metabolic sydrome column
# Display the column names to check if 'Metabolic_Syndrome' exists
print(mets.columns)

# Display the unique values in this column
if 'Metabolic_Syndrome' in mets.columns:
	print(mets['Metabolic_Syndrome'].unique())
else:
	print("Column 'Metabolic_Syndrome' does not exist in the dataframe.")

# Calculate value counts and their percentages
met_syd = mets['Metabolic_Syndrome'].value_counts()
met_syd_percentages = mets['Metabolic_Syndrome'].value_counts(normalize=True) * 100

# Combine counts and percentages into a single string
met_syd_summary = met_syd.astype(str) + " (" + met_syd_percentages.round(2).astype(str) + "%)"

# Display the result
print(met_syd_summary)

### Diabetis Mellitus column
# Display the unique values in this column
print(mets['Diabetes_Mellitus_status'].unique())

# Calculate value counts and their percentages
diabetes_counts = mets['Diabetes_Mellitus_status'].value_counts()
diabetes_percentages = mets['Diabetes_Mellitus_status'].value_counts(normalize=True) * 100

# Combine counts and percentages into a single string
diabetes_summary = diabetes_counts.astype(str) + " (" + diabetes_percentages.round(2).astype(str) + "%)"

# Display the result
print(diabetes_summary)

### Tobbaco use column
# Display the unique values in this column
print(mets['Tobbacco_Use'].unique())

# Calculate value counts and their percentages
tabacco_counts = mets['Tobbacco_Use'].value_counts()
tabacco_percentages = mets['Tobbacco_Use'].value_counts(normalize=True) * 100

# Combine counts and percentages into a single string
tabacco_summary = tabacco_counts.astype(str) + " (" + tabacco_percentages.round(2).astype(str) + "%)"

# Display the result
print(tabacco_summary)

# Alcohol use in column
# Display the unique values in this column
print(mets['Alcohol_Consuption'].unique())

# Calculate value counts and their percentages
alcohol_counts = mets['Alcohol_Consuption'].value_counts()
alcohol_percentages = mets['Alcohol_Consuption'].value_counts(normalize=True) * 100

# Combine counts and percentages into a single string
alcohol_summary = alcohol_counts.astype(str) + " (" + alcohol_percentages.round(2).astype(str) + "%)"

# Display the result
print(alcohol_summary)

# Level of Education column
# Display the unique values in this column
print(mets['Educational_Level'].unique())

# Calculate value counts and their percentages
education_counts = mets['Educational_Level'].value_counts()
education_percentages = mets['Educational_Level'].value_counts(normalize=True) * 100

# Combine counts and percentages into a single string
education_summary = education_counts.astype(str) + " (" + education_percentages.round(2).astype(str) + "%)"

# Display the result
print(education_summary)

# Numerical data types
# Display the descriptive statistics for the numeric columns
mets.describe().T.round(2)

# Exploratory Data Analysis
# Create a function to create a Histogram
def hist_plot(column_name, unit_of_measure, bin_count):
    fig, ax = plt.subplots(nrows=1, figsize=(8,4), facecolor='w')
    plt.title(column_name, fontsize = 22, weight='bold')
    sns.histplot(data=mets, x=column_name, color='#069AF3', 
                 linewidth=2, bins=bin_count); 
    plt.xlabel(unit_of_measure, fontsize = 16, weight='bold')
    plt.xticks(weight='bold')
    ax.set_ylabel('Instances',fontweight='bold',fontsize=14)
    ax.set_facecolor('lightblue')
    ax.tick_params(labelcolor='k', labelsize=10)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)

# Create a function to create a Histogram and Boxplot in the same figure
def hist_box_plot(column_name, unit_of_measure, bin_count):
    fig, (ax1,ax2) = plt.subplots(nrows =2, sharex=True, figsize=(8, 6), 
                                  facecolor='w', 
                                  gridspec_kw={'height_ratios':[0.75, 0.25]})
    plt.suptitle(f'{column_name}', y=1.02, va='center', 
                 fontsize = 22, weight='bold')
    sns.histplot(data=mets, x=column_name, color='#069AF3', linewidth=2, 
                 ax=ax1, bins=bin_count);
    plt.xlabel(unit_of_measure, fontsize = 16, weight='bold')
    plt.xticks(weight='bold')
    ax1.set_ylabel('Instances',fontweight='bold',fontsize=18)
    ax1.set_facecolor('lightblue')
    ax1.tick_params(labelcolor='k', labelsize=10)
    ax1.set_yticklabels(ax1.get_yticks(), weight='bold')
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(3)
    sns.boxplot(data=mets, x=column_name, width=.5, color='#069AF3', ax=ax2,
                medianprops={'color':'k', 'linewidth':2},
                whiskerprops={'color':'k', 'linewidth':2},
                boxprops={'facecolor':'#069AF3', 
                          'edgecolor':'k', 'linewidth':2},
                capprops={'color':'k', 'linewidth':3}); 
    ax2.tick_params(labelcolor='k', labelsize=10)
    ax2.set(xlabel=unit_of_measure);
    ax2.set_xticklabels(ax2.get_xticks(), fontweight='bold')
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['top'].set_color(None)
    ax2.spines['left'].set_color(None)
    ax2.spines['right'].set_color(None)
    plt.tight_layout();
    plt.show()

# Create a function to create a Count Plot
def count_plot(column_name, label_order):
    fig, ax = plt.subplots(nrows=1, figsize=(8,4), facecolor='w')
    plt.title(column_name, fontsize = 22, weight='bold')
    sns.countplot(data=mets, x=column_name, lw=3, ec='k', 
                  color='#069AF3', order=label_order)
    plt.xlabel('')
    plt.xticks(weight='bold')
    ax.set_ylabel('Instances', fontweight='bold', fontsize=18)
    ax.set_facecolor('lightblue')
    ax.tick_params(labelcolor='k', labelsize=12)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    plt.tight_layout()
    plt.show()

# Create a function to create a Count Plot
# Sex and Age correlation to MetabolicSyndrome
def scatter_plot(x,y, x_uom='', y_uom=''):
    palette_dict = {'No MetSyn': 'indigo', 'MetSyn': 'magenta'}
    fig, ax = plt.subplots(figsize=(8,4), facecolor='w')
    sns.scatterplot(x=x, y=y, hue="Metabolic_Syndrome", 
                    palette=palette_dict, data=mets);
    plt.title(f'{x} and {y} \ncorrelation to Metabolic Syndrome', fontsize = 18, weight='bold')
    plt.xlabel(f'{x} {x_uom}', fontsize = 14, weight='bold')
    plt.ylabel(f'{y} {y_uom}', fontsize = 14, weight='bold');
    plt.xticks(fontsize = 10, weight='bold')
    plt.yticks(fontsize = 10, weight='bold');
    ax.set_facecolor('lightblue')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    plt.legend(bbox_to_anchor=(1.23, 1))
    plt.tight_layout()
    plt.show()

## Numerical Columns

### summary statistics
# Display the descriptive statistics for the numeric columns
mets.describe().T.round(2)

# Display histogram charts for the numeric columns in the dataframe
mets.hist(figsize=(12,9), bins=30)
plt.tight_layout()
plt.show

# Display column statistics
mets['Age (yrs)'].describe().round(2)

# Display supplemental column statistics
column_statistics('Age (yrs)')

# Utilize function to display histogram plot with a custom title
hist_plot('Age (yrs)', 'years', 50)
plt.title('Age Distribution', fontsize=22, weight='bold')
plt.show()

# Display column statistics
mets['BMI(kg/m2)'].describe().round(2)

# Display supplemental column statistics
column_statistics('BMI(kg/m2)')

# Utilize function to display histogram plot with a custom title
hist_plot('BMI(kg/m2)', 'BMI(kg/m2)', 50)
plt.title('BMI Distribution', fontsize=22, weight='bold')
plt.show()

# Utilize function to display histogram and boxplot
hist_box_plot('BMI(kg/m2)', 'BMI(kg/m2)', 60)

# Display supplemental column statistics
column_statistics('Bp_Systolic (mmHg)')

# Utilize function to display histogram plot
hist_plot('Bp_Systolic (mmHg)', 'mmHg', 50)
plt.title('Blood Pressure systolic Distribution', fontsize=22, weight='bold')
plt.show()

# Utilize function to display histogram and boxplot
hist_box_plot('Bp_Systolic (mmHg)', 'mmHg', 60)

# Display supplemental column statistics
column_statistics('Bp_Diastolic (mmHg)')

# Utilize function to display histogram plot
hist_plot('Bp_Diastolic (mmHg)', 'mmHg', 50)
plt.title('Blood Pressure Diastolic Distribution', fontsize=22, weight='bold')
plt.show()

# Utilize function to display histogram and boxplot
hist_box_plot('Bp_Diastolic (mmHg)', 'mmHg', 60)

# Check if the column exists in the dataframe
if 'BP_Diastolic (mmHg)' in mets.columns:
	# Calculate the IQR for the "BP_Diastolic (mmHg)" column
	q1 = mets['BP_Diastolic (mmHg)'].quantile(0.25)
	q3 = mets['BP_Diastolic (mmHg)'].quantile(0.75)
	iqr = q3 - q1

	# Define the lower and upper bounds for outliers
	lower_bound = q1 - 1.5 * iqr
	upper_bound = q3 + 1.5 * iqr

	# Identify the outliers
	outliers = mets[(mets['BP_Diastolic (mmHg)'] < lower_bound) | (mets['BP_Diastolic (mmHg)'] > upper_bound)]

	# Display the outliers
	print(outliers)
else:
	print("Column 'BP_Diastolic (mmHg)' does not exist in the dataframe.")

# Rename Blood sugar to blood Glucose column
mets.rename(columns={'Blood_Glucose (mmol/L)': 'Blood_Glucose (mmol/L)'}, inplace=True)

# Display the updated column names to verify the change
print(mets.columns)

# Display column statistics
mets['Blood_Glucose (mmol/L)'].describe().round(2)

# Display supplemental column statistics
column_statistics('Blood_Glucose (mmol/L)')

# Utilize function to display histogram plot
hist_plot('Blood_Glucose (mmol/L)', 'mmol/L', 50)

# Utilize function to display histogram and boxplot
hist_box_plot('Blood_Glucose (mmol/L)', 'Glucose(mmol/L)', 50)

mets.describe().T.round(2)

# Display column statistics
mets['Triglycerides (mmol/L)'].describe().round(2)

# Display supplemental column statistics
column_statistics('Triglycerides (mmol/L)')

# Utilize function to display histogram plot
hist_plot('Triglycerides (mmol/L)', 'Trig(mmol/L)', 50)

# Utilize function to display histogram and boxplot
hist_box_plot('Triglycerides (mmol/L)', 'Trig(mmol/L)', 60)

# Display column statistics
mets['Cholesterol_HDL_(mmol/L)'].describe().round(2)

# Display supplemental column statistics
column_statistics('Cholesterol_HDL_(mmol/L)')

# Utilize function to display histogram plot
hist_plot('Cholesterol_HDL_(mmol/L)', 'HDL(mmol/L)', 50)

# Utilize function to display histogram and boxplot
hist_box_plot('Cholesterol_HDL_(mmol/L)', 'HDL(mmol/L)', 60)

# Calculate the IQR for the "Cholesterol_HDL_(mmol/L)" column
q1 = mets['Cholesterol_HDL_(mmol/L)'].quantile(0.25)
q3 = mets['Cholesterol_HDL_(mmol/L)'].quantile(0.75)
iqr = q3 - q1

# Define the lower and upper bounds for outliers
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Identify the outliers
outliers = mets[(mets['Cholesterol_HDL_(mmol/L)'] < lower_bound) | (mets['Cholesterol_HDL_(mmol/L)'] > upper_bound)]

# Display the outliers
print(outliers)

# Display the descriptive statistics for the non-numeric columns
mets.describe(exclude=('number'))

# Display column statistics
mets.Sex.describe()

# Display supplemental column statistics
column_statistics('Sex')

# Display normailzed value counts
mets['Sex'].value_counts(normalize=True).round(2)

# Utilize function to display count plot
count_plot('Sex', ['female', 'male'])

# Display column statistics
mets['Regimen_Type'].describe()

# Calculate value counts and their percentages
drugcode_counts = mets['Regimen_Type'].value_counts()
drugcode_percentages = mets['Regimen_Type'].value_counts(normalize=True) * 100

# Combine counts and percentages into a single string
drugcode_summary = drugcode_counts.astype(str) + " (" + drugcode_percentages.round(2).astype(str) + "%)"

# Display the result
print(drugcode_summary)

# Utilize function to display count plot
count_plot('Regimen_Type', ['TAFED', 'TLD', 'AZT+3TC+ATVr', 'AZT+3TC+LPVr'])

# Calculate value counts and their percentages
met_syd_counts = mets['Metabolic_Syndrome'].value_counts()
met_syd_percentages = mets['Metabolic_Syndrome'].value_counts(normalize=True) * 100

# Combine counts and percentages into a single string
met_syd_summary = met_syd_counts.astype(str) + " (" + met_syd_percentages.round(2).astype(str) + "%)"

# Display the result
print(met_syd_summary)

# Utilize function to display count plot
count_plot('Metabolic_Syndrome', ['MetSyn', 'No MetSyn'])

# Create a count plot for males and females with metabolic syndrome
sns.countplot(data=mets, x='Sex', hue='Metabolic_Syndrome', palette='Set1')
plt.title('Count of Males and Females with Metabolic Syndrome', fontsize=16, weight='bold')
plt.xlabel('Gender', fontsize=14, weight='bold')
plt.ylabel('Count', fontsize=14, weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.legend(title='Metabolic Syndrome', title_fontsize='13', fontsize='12')
plt.show()

sns.countplot(y='Regimen_Type', hue='Metabolic_Syndrome', data = mets)

# Calculate value counts and their percentages
education_counts = mets['Educational_Level'].value_counts()
education_percentages = mets['Educational_Level'].value_counts(normalize=True) * 100

# Combine counts and percentages into a single string
education_summary = education_counts.astype(str) + " (" + education_percentages.round(2).astype(str) + "%)"

# Display the result
print(education_summary)

# Utilize function to display count plot
count_plot('Educational_Level', ['primary', 'secondary', 'Terciary', 'Unknown', 'none'])

# Display Display column column statistics statistics
mets['Alcohol_Consuption'].describe()

# Calculate the total number of Alcohol_use
total_alcohol_use = alcohol_counts.sum()

# Format the summary with brackets in percentage
alcohol_summary = alcohol_counts.astype(str) + " (" + alcohol_percentages.round(2).astype(str) + "%)"

# Display the total number and the formatted summary
print(f"Total Alcohol Use: {total_alcohol_use}")
print(alcohol_summary)

# Utilize function to display count plot
count_plot('Alcohol_Consuption', ['no', 'yes', 'Unknown'])

# Display column statistics
mets['Tobbacco_Use'].describe()

# Calculate the total number of Tabbacco_use
total_tabbacco_use = tabacco_counts.sum()

# Format the summary with brackets in percentage
tabacco_summary = tabacco_counts.astype(str) + " (" + tabacco_percentages.round(2).astype(str) + "%)"

# Display the total number and the formatted summary
print(f"Total Tabbacco Use: {total_tabbacco_use}")
print(tabacco_summary)

# Check the column names to confirm the correct spelling
print(mets.columns)

# Utilize function to display count plot
count_plot('Tobbacco_Use', ['no', 'yes'])

# Display diabetis Mellitus status column statistics
mets['Diabetes_Mellitus_status'].describe()

# Calculate the total number of diabetes_mellitus_status
total_diabetes_status = diabetes_counts.sum()

# Format the summary with brackets in percentage
diabetes_summary = diabetes_counts.astype(str) + " (" + diabetes_percentages.round(2).astype(str) + "%)"

# Display the total number and the formatted summary
print(f"Total Diabetes Mellitus Status: {total_diabetes_status}")
print(diabetes_summary)

# Utilize function to display count plot
count_plot('Diabetes_Mellitus_status', ['no', 'yes', 'Unknown'])

### Explanatory Data Analysis
### Scatter plot of BMI vs Waist Circumference and Metabolyc Syndrome
sns.scatterplot(data = mets, x = 'BMI(kg/m2)', y = 'Waist_Circumference(cm)', hue = 'Metabolic_Syndrome')
plt.legend();

### Scatter plot of Blood Glucose vs Triglycerides and Metabolyc_Syndrome
sns.scatterplot(data = mets, x = 'Blood_Glucose (mmol/L)', y = 'Triglycerides (mmol/L)', hue = 'Metabolic_Syndrome')
plt.legend();

#### WaistCirc and Age correlation to MetabolicSyndrome
scatter_plot('Waist_Circumference(cm)', 'Age (yrs)')

### BMI and Age correlation to Metabolic Syndrome
scatter_plot('BMI(kg/m2)', 'Age (yrs)')

### Cd4 Count and Age Correction to metabolic syndrome
scatter_plot('CD4_count(cells/µl)', 'Age (yrs)')

### Regimen_Type and Age correlation to Metabolic Syndrome
scatter_plot('Regimen_Type', 'Age (yrs)')

### Distribution of metabolic syndrome by regimem types
sns.countplot(data=mets, x='Regimen_Type', hue='Metabolic_Syndrome', palette='Set2')
plt.title('Distribution of Metabolic Syndrome by Regimen Type', fontsize=16, weight='bold')
plt.xlabel('Regimen Type', fontsize=14, weight='bold')
plt.ylabel('Count', fontsize=14, weight='bold')
plt.xticks(rotation=45, weight='bold')
plt.yticks(weight='bold')
plt.legend(title='Metabolic Syndrome', title_fontsize='13', fontsize='12')
plt.tight_layout()
plt.show()

# Group by 'Regimen_Type' and 'Metabolic_Syndrome' and count the occurrences
regimen_met_syn_counts = mets.groupby(['Regimen_Type', 'Metabolic_Syndrome']).size().unstack(fill_value=0)

# Display the result
print(regimen_met_syn_counts)

### Gender and Age(years) correction to metabolic syndrome
scatter_plot('Sex', 'Age (yrs)')

### Distribution of metabolic syndrome by Gender
sns.countplot(data=mets, x='Sex', hue='Metabolic_Syndrome', palette='Set2')
plt.title('Distribution of Metabolic Syndrome by Sex', fontsize=16, weight='bold')
plt.xlabel('Sex', fontsize=14, weight='bold')
plt.ylabel('Count', fontsize=14, weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.legend(title='Metabolic Syndrome', title_fontsize='13', fontsize='12')
plt.show()

### Diabetes Mellitus status and Age(years) correction to metabolic syndrome
scatter_plot('Diabetes_Mellitus_status', 'Age (yrs)')

# Create a count plot for metabolic syndrome and diabetes status
sns.countplot(data=mets, x='Diabetes_Mellitus_status', hue='Metabolic_Syndrome', palette='Set2')
plt.title('Distribution of Metabolic Syndrome with Diabetes Status', fontsize=16, weight='bold')
plt.xlabel('Diabetes Mellitus Status', fontsize=14, weight='bold')
plt.ylabel('Count', fontsize=14, weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.legend(title='Metabolic Syndrome', title_fontsize='13', fontsize='12')
plt.show()

### Tobbaco Use and Age correction to Metabolic syndrome
scatter_plot('Tobbacco_Use', 'Age (yrs)')

### Alcohol use and Age Correction to metabolic syndrome
scatter_plot('Alcohol_Consuption', 'Age (yrs)')

### Distribution of metabolic syndrome by alcohol Consumption
sns.countplot(data=mets, x='Alcohol_Consuption', hue='Metabolic_Syndrome', palette='Set2')
plt.title('Distribution of Metabolic Syndrome with Alcohol Consumption', fontsize=16, weight='bold')
plt.xlabel('Alcohol Consumption', fontsize=14, weight='bold')
plt.ylabel('Count', fontsize=14, weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.legend(title='Metabolic Syndrome', title_fontsize='13', fontsize='12')
plt.show()

### Level of Education and Age correction to metabolic syndrome
scatter_plot('Educational_Level', 'Age (yrs)')

# Create a count plot for educational level and metabolic syndrome
sns.countplot(data=mets, x='Educational_Level', hue='Metabolic_Syndrome', palette='Set2')
plt.title('Distribution of Metabolic Syndrome by Educational Level', fontsize=16, weight='bold')
plt.xlabel('Educational Level', fontsize=14, weight='bold')
plt.ylabel('Count', fontsize=14, weight='bold')
plt.xticks(rotation=45, weight='bold')
plt.yticks(weight='bold')
plt.legend(title='Metabolic Syndrome', title_fontsize='13', fontsize='12')
plt.show()

### Blood sugar and age correction to Metabolic Syndrome
scatter_plot('Blood_Glucose (mmol/L)', 'Age (yrs)')

# Create a count plot for metabolic syndrome and event name
sns.countplot(data=mets, x='Event_Name', hue='Metabolic_Syndrome', palette='Set2')
plt.title('Distribution of Metabolic Syndrome by Event Name', fontsize=16, weight='bold')
plt.xlabel('Event Name', fontsize=14, weight='bold')
plt.ylabel('Count', fontsize=14, weight='bold')
plt.xticks(rotation=45, weight='bold')
plt.yticks(weight='bold')
plt.legend(title='Metabolic Syndrome', title_fontsize='13', fontsize='12')
plt.tight_layout()
plt.show()

# Calculate the prevalence of metabolic syndrome per event name
prevalence = mets.groupby('Event_Name')['Metabolic_Syndrome'].value_counts(normalize=True).unstack().fillna(0) * 100

# Round the prevalence to two decimal places
prevalence = prevalence.round(2)

# Sort the prevalence from lowest to highest
sorted_prevalence = prevalence.sort_values(by='MetSyn')

# Display the result
print(sorted_prevalence)

# Calculate the prevalence of metabolic syndrome per event name
prevalence = mets.groupby('Event_Name')['Metabolic_Syndrome'].value_counts(normalize=True).unstack().fillna(0) * 100

# Round the prevalence to two decimal places
prevalence = prevalence.round(2)

# Sort the prevalence from lowest to highest
sorted_prevalence = prevalence.sort_values(by='MetSyn')

# Plot the sorted prevalence
sorted_prevalence['MetSyn'].plot(kind='bar', color='navy')
plt.title('Prevalence of Metabolic Syndrome by Event Name', fontsize=16, weight='bold')
plt.xlabel('Event Name', fontsize=14, weight='bold')
plt.ylabel('Prevalence (%)', fontsize=14, weight='bold')
plt.xticks(rotation=45, weight='bold')
plt.yticks(weight='bold')
plt.tight_layout()
plt.show()

### Triglycerides and Age correction to metabolic syndrome
scatter_plot('Triglycerides (mmol/L)', 'Age (yrs)')

### Cholesterol HDL and Age Correlation to Metabolic syndrome
scatter_plot('Cholesterol_HDL_(mmol/L)', 'Age (yrs)')

### Viral load and Age Correction to Metabolic syndrome
scatter_plot('Viral_Load(cp/ml)', 'Age (yrs)')

### Correlation Heatmap
# Define a dictionary with key/value pairs and use it to replace values
dict = {"No MetSyn": 0, "MetSyn": 1}
mets.replace({'Metabolic_Syndrome': dict}, inplace = True)
mets.Metabolic_Syndrome.astype('int32').dtypes

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 10), facecolor='w')
# Select only numeric columns for correlation calculation
numeric_cols = mets.select_dtypes(include=['number'])
corr = numeric_cols.corr()
sns.heatmap(corr, cmap='viridis', annot=True)
plt.title('Correlation Heatmap', fontsize=24, weight='bold')
plt.xticks(fontsize=14, weight='bold', rotation=90)
plt.yticks(fontsize=14, weight='bold', rotation=0)
plt.tight_layout()
plt.show()

## Preprocessing/Processing for Machine Learning
- ordinal features = [Educational Levell]
- numeric features = ['Age', 'Waist circumference', 'BMI', 'Blood sugar, cholesterol HDL,Blood Glucose, Triglycerides]
- nominal features = ['Sex', 'MetabolicSyndrome', 'Drug_Code',  'Diabetes_Mellitus_status', 'Tobbacco_Use', 'Alcohol_Consuption']
- date/time features = Event Name
- pass through = none

### Evaluate if Target is Balanced
# Display normalized target value counts
mets['Metabolic_Syndrome'].value_counts(normalize=True)

### Validation Split
# Define features (X) and target (y)
X = mets.drop(columns = ['Metabolic_Syndrome',])
y = mets['Metabolic_Syndrome']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

### Instantiate Column Selectors
# Create column selectors.
cat_selector = make_column_selector(dtype_include='object')
num_selector = make_column_selector(dtype_include='number')

# Display the list of categorical column names.
cat_selector(X_train)

# Display the list of numeric column names.
num_selector(X_train)

### Instantiate Transformers
# Imputers
freq_imputer = SimpleImputer(strategy='most_frequent')
median_imputer = SimpleImputer(strategy='median')
# Scaler
scaler = StandardScaler()
# One Hot Encoder
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

### Instantiate Pipelines
# Create the numeric pipeline
numeric_pipe = make_pipeline(median_imputer, scaler)
# Display the numeric pipeline
numeric_pipe

# Create the categorical pipeline
categorical_pipe = make_pipeline(freq_imputer, ohe)
# Display the categorical pipeline
categorical_pipe

### Instantiate ColumnTransformer
# Create tuples for the Column Transformers
number_tuple = (numeric_pipe, num_selector)
category_tuple = (categorical_pipe, cat_selector)
# Create the ColumnTransformer
preprocessor = make_column_transformer(number_tuple, 
                                       category_tuple, 
                                       remainder='passthrough')
# Display the ColumnTransformer
preprocessor

### Fit and Transform Data
# Fit on Train
preprocessor.fit(X_train)

# Transform Train and Test
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Check for missing values and that data is scaled and one-hot encoded
print(np.isnan(X_train_processed).sum().sum(), 'missing values in training data')
print(np.isnan(X_test_processed).sum().sum(), 'missing values in testing data')
print('\n')
print('All data in X_train_processed are', X_train_processed.dtype)
print('All data in X_test_processed are', X_test_processed.dtype)
print('\n')
print('shape of data is', X_train_processed.shape)
print('\n')
X_train_processed

X_test_df = pd.DataFrame(X_test_processed)
X_train_df = pd.DataFrame(X_train_processed)

X_train_df.describe().round(2)
X_test_df.describe().round(2)

### KNN Model
# Make an instance of the Model
knn = KNeighborsClassifier()

# Create Pipeline
knn_model_processor = make_pipeline(knn)

# Fit on Train
knn_model_processor.fit(X_train_processed, y_train)

# Calculate Model Accuracy
knn_model_accuracy = accuracy_score(y_test, knn_model_processor.predict(X_test_processed))

# Display Model Accuracy
print(f'KNN Model Accuracy = {knn_model_accuracy}')

# Display the hyperparameters.
knn_model_processor.get_params()

## Tune with loop
# Tune K utilizing a loop.
krange = range(1, 20)
k_list = []
for k in krange: 
  knn_model_processor = make_pipeline(KNeighborsClassifier(n_neighbors=k))
  knn_model_processor.fit(X_train_processed, y_train)
  score = accuracy_score(y_test, knn_model_processor.predict(X_test_processed))
  k_list.append(score)

# Visualize Accuracy Scores.
plt.plot(krange, k_list)
plt.xlabel('K')
plt.ylabel('Score');

# Create Pipeline utilizing hyperparameters with highest accuracy
knn_model_processor = make_pipeline(KNeighborsClassifier(n_neighbors=11))
# Fit on Train
knn_model_processor.fit(X_train_processed, y_train)

# Print Model Accuracy Score
knn_model_accuracy = accuracy_score(y_test, knn_model_processor.predict(X_test_processed))
print(f'KNN Model Accuracy Score = {knn_model_accuracy}')

### Tune using GridSearch
### Select Hyperparameters
# Select hyperparameters
knn_parameters = { 'kneighborsclassifier__n_neighbors': [8,9,10,11,12,13,14], 
                  'kneighborsclassifier__leaf_size': [2, 3, 4, 5,10,15,20,25]}

### Instantiate Grid Search
# Instantiate Gridsearch
knn_grid = GridSearchCV(knn_model_processor, knn_parameters)
# Fit GridSearch
knn_grid.fit(X_train_processed, y_train)

# Display the best hyperparameters From GridSearchCV.
knn_grid.best_params_

# Extract KNN Model with best hyperparameters.
best_knn = knn_grid.best_estimator_

## Predictions
# Predictions from KNN Model with the best hyperparameters.
knn_train_preds = best_knn.predict(X_train_processed)
knn_test_preds = best_knn.predict(X_test_processed)

## Scores
## Accuracy scores
# Calculate classification accuracy scores
knn_train_accuracy = best_knn.score(X_train_processed, y_train)
knn_test_accuracy = best_knn.score(X_test_processed, y_test)
# Print classification accuracy scores
print(f'KNN Train Accuracy Score: {knn_train_accuracy}')
print(f'KNN Test Accuracy Score: {knn_test_accuracy}')

## Recall score
# Calculate classification recall scores
knn_train_recall = recall_score(y_train, knn_train_preds, pos_label=1)
knn_test_recall = recall_score(y_test, knn_test_preds, pos_label=1)
# Print classification recall scores
print(f'KNN Train Recall Score = {knn_train_recall}')
print(f'KNN Test Recall Score = {knn_test_recall}')

### Precision Scores
# Calculate classification precision scores
knn_train_precision = precision_score(y_train, knn_train_preds, pos_label=1)
knn_test_precision = precision_score(y_test, knn_test_preds, pos_label=1)

# Print precision scores
print(f'KNN Train Precision Score = {knn_train_precision}')
print(f'KNN Test Precision Score = {knn_test_precision}')

#AUC scores
# Calculate AUC scores
knn_train_auc = roc_auc_score(y_train, best_knn.predict_proba(X_train_processed)[:,1])
knn_test_auc = roc_auc_score(y_test, best_knn.predict_proba(X_test_processed)[:,1])
# Display AUC scores
print(f'KNN Train AUC: {knn_train_auc}')
print(f'KNN Test AUC: {knn_test_auc}')

## Logistic Regression Model
# instantiate Logistic Regression Model
# Make an instance of the model
logreg = LogisticRegression(C = 1000)

### Create the Pipeline
# Create pipeline
logreg_pipe = make_pipeline(logreg)

### Fit and Train the Model on the Data
# Fit the model
logreg_pipe.fit(X_train_processed,y_train)

### Print Training Scores
# Fit the model (ensure the model is fitted before scoring)
logreg_pipe.fit(X_train_processed, y_train)

# Print Scores
print(logreg_pipe.score(X_train_processed, y_train))
print(logreg_pipe.score(X_test_processed, y_test))

### Hyperparameter Tuning
### L1 Tuning
### Wideband Tuning
# Create a list of C values and empty lists for accuracy scores
c_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_scores = []
test_scores = []
 
# Create a loop to iterative over the C values list
for c in c_values:
 
  # Instantiate and Fit the Model on the data
  log_reg = LogisticRegression(C=c, 
                               max_iter=1000, 
                               solver='liblinear', 
                               penalty='l1')
  log_reg_pipe = make_pipeline(log_reg)
  log_reg_pipe.fit(X_train_processed, y_train)
 
  # Add the Train and Test Scores to our Scores Lists
  train_scores.append(log_reg_pipe.score(X_train_processed, y_train))
  test_scores.append(log_reg_pipe.score(X_test_processed, y_test))
 
# Plot the Accuracy Scores for the C Values
fig, ax = plt.subplots(1,1)
ax.plot(c_values, train_scores, label='Training Accuracy')
ax.plot(c_values, test_scores, label='Testing Accuracy')
ax.set_xticks(c_values)
ax.set_title('Change in accuracy over C values for l1 regularization')
ax.legend()
 
# Set the X Axis to a logarithmic Scale
ax.set_xscale('log')

# Print a Dictionary for C Values and Accuracy Scores
{c:score for c, score in zip(c_values, test_scores)}

### Fine Tuning
# Create a list of C values and empty lists for accuracy scores
c_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
train_scores = []
test_scores = []
 
# Create a loop to iterative over the C values list
for c in c_values:
 
  # Instantiate and Fit the Model on the data
  log_reg = LogisticRegression(C=c, 
                               max_iter=1000, 
                               solver='liblinear', 
                               penalty='l1')
  log_reg_pipe = make_pipeline(log_reg)
  log_reg_pipe.fit(X_train_processed, y_train)
 
  # Add the Train and Test Scores to our Scores Lists
  train_scores.append(log_reg_pipe.score(X_train_processed, y_train))
  test_scores.append(log_reg_pipe.score(X_test_processed, y_test))
 
# Plot the Accuracy Scores for the C Values
fig, ax = plt.subplots(1,1)
ax.plot(c_values, train_scores, label='Training Accuracy')
ax.plot(c_values, test_scores, label='Testing Accuracy')
ax.set_xticks(c_values)
ax.set_title('Change in accuracy over C values for l1 regularization')
ax.legend()
 
# Set the X Axis to a logarithmic Scale
ax.set_xscale('log')

# Print a Dictionary for C Values and Accuracy Scores
{c:score for c, score in zip(c_values, test_scores)}

### Best L2 Tuned Logistics Regression Model
best_l2_log_reg = LogisticRegression(C=.020, 
                                     max_iter=1000, 
                                     solver='liblinear', 
                                     penalty='l2')
best_l2_log_reg_pipe = make_pipeline(best_l2_log_reg)
best_l2_log_reg_pipe.fit(X_train_processed, y_train)

### Predictions
# Predictions from Best L2 Tuned Logistics Regression Model
log_reg_l2_train_preds = best_l2_log_reg_pipe.predict(X_train_processed)
log_reg_l2_test_preds = best_l2_log_reg_pipe.predict(X_test_processed)

### Scores
### Accuracy Scores
# Calculate classification accuracy scores
log_reg_l2_train_accuracy = best_l2_log_reg_pipe.score(X_train_processed, y_train)
log_reg_l2_test_accuracy = best_l2_log_reg_pipe.score(X_test_processed, y_test)
# Print classification accuracy scores
print(f'Logistics Regression L2 tuned Train Accuracy Score: {log_reg_l2_train_accuracy}')
print(f'Logistics Regression L2 tuned Test Accuracy Score: {log_reg_l2_test_accuracy}')

### Recall Scores
# Calculate classification recall scores
log_reg_l2_train_recall = recall_score(y_train, log_reg_l2_train_preds, pos_label=1)
log_reg_l2_test_recall = recall_score(y_test, log_reg_l2_test_preds, pos_label=1)
# Print recall scores
print(f'Logistics Regression L2 tuned Train Recall Score: {log_reg_l2_train_recall}')
print(f'Logistics Regression L2 tuned Test Recall Score: {log_reg_l2_test_recall}')

### Precision Scores
# Calculate classification recall scores
log_reg_l2_train_precision = precision_score(y_train, log_reg_l2_train_preds, pos_label=1)
log_reg_l2_test_precision = precision_score(y_test, log_reg_l2_test_preds, pos_label=1)
# Display recall scores
print(f'Logistics Regression L2 tuned Train Precision Score: {log_reg_l2_train_precision:.4f}}}')
print(f'Logistics Regression L2 tuned Test Precision Score: {log_reg_l2_test_precision:.4f}}}')

## AUC scores
# Calculate AUC scores
log_reg_l2_train_auc = roc_auc_score(y_train, best_l2_log_reg_pipe.predict_proba(X_train_processed)[:,1])
log_reg_l2_test_auc = roc_auc_score(y_test, best_l2_log_reg_pipe.predict_proba(X_test_processed)[:,1])
# Display AUC scores
print(f'Logistics Regression L2 tuned Train AUC Score: {log_reg_l2_train_auc:.4f}')
print(f'Logistics Regression L2 tuned Test AUC Score: {log_reg_l2_test_auc:.4f}')

# Decision Tree
### Instantiate the Decision Tree
# Instantiate the Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)

### Fit and Train the Model on the Data
# Fit the model
decision_tree.fit(X_train_processed, y_train)

# Print Scores
print(decision_tree.score(X_train_processed, y_train))
print(decision_tree.score(X_test_processed, y_test))

### Accuracy score
# Make predictions
dt_train_preds = decision_tree.predict(X_train_processed)
dt_test_preds = decision_tree.predict(X_test_processed)

# Accuracy Scores
dt_train_accuracy = accuracy_score(y_train, dt_train_preds)
dt_test_accuracy = accuracy_score(y_test, dt_test_preds)
print(f'Decision Tree Train Accuracy Score: {dt_train_accuracy}')
print(f'Decision Tree Test Accuracy Score: {dt_test_accuracy}')

### Recall prediction
# Recall Scores
dt_train_recall = recall_score(y_train, dt_train_preds, pos_label=1)
dt_test_recall = recall_score(y_test, dt_test_preds, pos_label=1)
print(f'Decision Tree Train Recall Score: {dt_train_recall}')
print(f'Decision Tree Test Recall Score: {dt_test_recall}')

### Precision score
# Precision Scores
dt_train_precision = precision_score(y_train, dt_train_preds, pos_label=1)
dt_test_precision = precision_score(y_test, dt_test_preds, pos_label=1)
print(f'Decision Tree Train Precision Score: {dt_train_precision}')
print(f'Decision Tree Test Precision Score: {dt_test_precision}')

### AUC scores
# AUC Scores
dt_train_auc = roc_auc_score(y_train, decision_tree.predict_proba(X_train_processed)[:,1])
dt_test_auc = roc_auc_score(y_test, decision_tree.predict_proba(X_test_processed)[:,1])
print(f'Decision Tree Train AUC Score: {dt_train_auc}')
print(f'Decision Tree Test AUC Score: {dt_test_auc}')

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

# Instantiate the Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)

# Fit the model
decision_tree.fit(X_train_processed, y_train)

# Predictions
dt_train_preds = decision_tree.predict(X_train_processed)
dt_test_preds = decision_tree.predict(X_test_processed)

# Accuracy Scores
dt_train_accuracy = accuracy_score(y_train, dt_train_preds)
dt_test_accuracy = accuracy_score(y_test, dt_test_preds)
print(f'Decision Tree Train Accuracy Score: {dt_train_accuracy}')
print(f'Decision Tree Test Accuracy Score: {dt_test_accuracy}')

# Recall Scores
dt_train_recall = recall_score(y_train, dt_train_preds, pos_label=1)
dt_test_recall = recall_score(y_test, dt_test_preds, pos_label=1)
print(f'Decision Tree Train Recall Score: {dt_train_recall}')
print(f'Decision Tree Test Recall Score: {dt_test_recall}')

# Precision Scores
dt_train_precision = precision_score(y_train, dt_train_preds, pos_label=1)
dt_test_precision = precision_score(y_test, dt_test_preds, pos_label=1)
print(f'Decision Tree Train Precision Score: {dt_train_precision}')
print(f'Decision Tree Test Precision Score: {dt_test_precision}')

# AUC Scores
dt_train_auc = roc_auc_score(y_train, decision_tree.predict_proba(X_train_processed)[:,1])
dt_test_auc = roc_auc_score(y_test, decision_tree.predict_proba(X_test_processed)[:,1])
print(f'Decision Tree Train AUC Score: {dt_train_auc}')
print(f'Decision Tree Test AUC Score: {dt_test_auc}')

### Gaussian naive Bayes Model
from sklearn.naive_bayes import GaussianNB

# Instantiate the Gaussian Naive Bayes model
gnb = GaussianNB()

# Fit the model
gnb.fit(X_train_processed, y_train)

# Predictions
gnb_train_preds = gnb.predict(X_train_processed)
gnb_test_preds = gnb.predict(X_test_processed)

# Accuracy Scores
gnb_train_accuracy = accuracy_score(y_train, gnb_train_preds)
gnb_test_accuracy = accuracy_score(y_test, gnb_test_preds)
print(f'Gaussian Naive Bayes Train Accuracy Score: {gnb_train_accuracy}')
print(f'Gaussian Naive Bayes Test Accuracy Score: {gnb_test_accuracy}')

# Recall Scores
gnb_train_recall = recall_score(y_train, gnb_train_preds, pos_label=1)
gnb_test_recall = recall_score(y_test, gnb_test_preds, pos_label=1)
print(f'Gaussian Naive Bayes Train Recall Score: {gnb_train_recall}')
print(f'Gaussian Naive Bayes Test Recall Score: {gnb_test_recall}')

# Precision Scores
gnb_train_precision = precision_score(y_train, gnb_train_preds, pos_label=1)
gnb_test_precision = precision_score(y_test, gnb_test_preds, pos_label=1)
print(f'Gaussian Naive Bayes Train Precision Score: {gnb_train_precision}')
print(f'Gaussian Naive Bayes Test Precision Score: {gnb_test_precision}')

# AUC Scores
gnb_train_auc = roc_auc_score(y_train, gnb.predict_proba(X_train_processed)[:,1])
gnb_test_auc = roc_auc_score(y_test, gnb.predict_proba(X_test_processed)[:,1])
print(f'Gaussian Naive Bayes Train AUC Score: {gnb_train_auc}')
print(f'Gaussian Naive Bayes Test AUC Score: {gnb_test_auc}')

### Support vector Machine
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

# Instantiate the Support Vector Machine model
svm = SVC(probability=True, random_state=42)

# Fit the model
svm.fit(X_train_processed, y_train)

# Predictions
svm_train_preds = svm.predict(X_train_processed)
svm_test_preds = svm.predict(X_test_processed)

# Accuracy Scores
svm_train_accuracy = accuracy_score(y_train, svm_train_preds)
svm_test_accuracy = accuracy_score(y_test, svm_test_preds)
print(f'SVM Train Accuracy Score: {svm_train_accuracy}')
print(f'SVM Test Accuracy Score: {svm_test_accuracy}')

# Recall Scores
svm_train_recall = recall_score(y_train, svm_train_preds, pos_label=1)
svm_test_recall = recall_score(y_test, svm_test_preds, pos_label=1)
print(f'SVM Train Recall Score: {svm_train_recall}')
print(f'SVM Test Recall Score: {svm_test_recall}')

# Precision Scores
svm_train_precision = precision_score(y_train, svm_train_preds, pos_label=1)
svm_test_precision = precision_score(y_test, svm_test_preds, pos_label=1)
print(f'SVM Train Precision Score: {svm_train_precision}')
print(f'SVM Test Precision Score: {svm_test_precision}')

# AUC Scores
svm_train_auc = roc_auc_score(y_train, svm.predict_proba(X_train_processed)[:,1])
svm_test_auc = roc_auc_score(y_test, svm.predict_proba(X_test_processed)[:,1])
print(f'SVM Train AUC Score: {svm_train_auc}')
print(f'SVM Test AUC Score: {svm_test_auc}')

### Extreme gradient boosting Model
from xgboost import XGBClassifier
# Convert 'MetSyn'/'No MetSyn' to 1/0 and handle NaN values
y_train_binary = y_train.map({'MetSyn': 1, 'No MetSyn': 0}).fillna(0)
y_test_binary = y_test.map({'MetSyn': 1, 'No MetSyn': 0}).fillna(0)

# Instantiate the XGBoost model
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Fit the model
xgb.fit(X_train_processed, y_train_binary)

# Predictions
xgb_train_preds = xgb.predict(X_train_processed)
xgb_test_preds = xgb.predict(X_test_processed)

# Accuracy Scores
xgb_train_accuracy = accuracy_score(y_train_binary, xgb_train_preds)
xgb_test_accuracy = accuracy_score(y_test_binary, xgb_test_preds)
print(f'XGBoost Train Accuracy Score: {xgb_train_accuracy}')
print(f'XGBoost Test Accuracy Score: {xgb_test_accuracy}')

# Recall Scores
xgb_train_recall = recall_score(y_train_binary, xgb_train_preds, pos_label=1)
xgb_test_recall = recall_score(y_test_binary, xgb_test_preds, pos_label=1)
print(f'XGBoost Train Recall Score: {xgb_train_recall}')
print(f'XGBoost Test Recall Score: {xgb_test_recall}')

# Precision Scores
xgb_train_precision = precision_score(y_train_binary, xgb_train_preds, pos_label=1)
xgb_test_precision = precision_score(y_test_binary, xgb_test_preds, pos_label=1)
print(f'XGBoost Train Precision Score: {xgb_train_precision}')
print(f'XGBoost Test Precision Score: {xgb_test_precision}')

# AUC Scores
xgb_train_auc = roc_auc_score(y_train_binary, xgb.predict_proba(X_train_processed)[:,1])
xgb_test_auc = roc_auc_score(y_test_binary, xgb.predict_proba(X_test_processed)[:,1])
print(f'XGBoost Train AUC Score: {xgb_train_auc}')
print(f'XGBoost Test AUC Score: {xgb_test_auc}')

### Random Forest Model
### Instantiate the Random Forest Model
# Instantiate the Random Forest model
ran_for = RandomForestClassifier(random_state=42)

# Fit the model
ran_for.fit(X_train_processed, y_train)

### Fit and Train the Model on the Data
# Fit the model
ran_for.fit(X_train_processed, y_train)

# Make an instance of the Model
ran_for = RandomForestClassifier(random_state = 42)

### Predictions
# Fit the RandomForestClassifier on the training data
ran_for.fit(X_train_processed, y_train)

# Predictions from RandomForestClassifier
ran_for_train_preds = ran_for.predict(X_train_processed)
ran_for_test_preds = ran_for.predict(X_test_processed)

# Scores
# Accuracy Scores for RF
# Calculate classification accuracy scores
ran_for_train_accuracy = ran_for.score(X_train_processed, y_train)
ran_for_test_accuracy = ran_for.score(X_test_processed, y_test)
# Print classification accuracy scores
print(f'Random Forest Train Accuracy Score: {ran_for_train_accuracy}')
print(f'Random Forest Test Accuracy Score: {ran_for_test_accuracy}')

### Recall scores for RF
# Calculate classification recall scores
ran_for_train_recall = recall_score(y_train, ran_for_train_preds, pos_label=1)
ran_for_test_recall = recall_score(y_test, ran_for_test_preds, pos_label=1)
# Print classification recall scores
print(f'Random Forest Train Recall Score = {ran_for_train_recall}')
print(f'Random Forest Test Recall Score = {ran_for_test_recall}')

### Precision scores for RF
# Calculate classification precision scores
ran_for_train_precision = precision_score(y_train, ran_for_train_preds, pos_label=1)
ran_for_test_precision = precision_score(y_test, ran_for_test_preds, pos_label=1)
# Print classification recall scores
print(f'Random Forest Train Precision Score = {ran_for_train_precision}')
print(f'Random Forest Test Precision Score = {ran_for_test_precision}')

### AUC for RF
# Calculate AUC scores
ran_for_train_auc = roc_auc_score(y_train, ran_for.predict_proba(X_train_processed)[:,1])
ran_for_test_auc = roc_auc_score(y_test, ran_for.predict_proba(X_test_processed)[:,1])
# Display AUC scores
print(f'Random Forest Train AUC: {ran_for_train_auc}')
print(f'Random Forest Test AUC: {ran_for_test_auc}')

### Model Comparision
### confusion Matrix

# Make an instance of the best L1 tuned Logistic Regression model
best_l1_log_reg = LogisticRegression(C=0.1, max_iter=1000, solver='liblinear', penalty='l1')
best_l1_log_reg_pipe = make_pipeline(best_l1_log_reg)
best_l1_log_reg_pipe.fit(X_train_processed, y_train)

# Predictions from Best L1 Tuned Logistics Regression Model
log_reg_l1_train_preds = best_l1_log_reg_pipe.predict(X_train_processed)
log_reg_l1_test_preds = best_l1_log_reg_pipe.predict(X_test_processed)

# Calculate the Confusion Matrices
knn_confusion_matrix = confusion_matrix(y_test, knn_test_preds)
l1_log_reg_confusion_matrix = confusion_matrix(y_test, log_reg_l1_test_preds)
l2_log_reg_confusion_matrix = confusion_matrix(y_test, log_reg_l2_test_preds)
ran_for_confusion_matrix = confusion_matrix(y_test, ran_for_test_preds)

# Display the Confusion Matrices
print(f'KNN Confusion Matrix:\n {knn_confusion_matrix}')
print('\n')
print(f'L1 Tuned Logistics Regression Confusion Matrix:\n {l1_log_reg_confusion_matrix}')
print('\n')
print(f'L2 Tuned Logistics Regression Confusion Matrix:\n {l2_log_reg_confusion_matrix}')
print('\n')
print(f'Random Forest Confusion Matrix:\n {ran_for_confusion_matrix}')

# Convert 'MetSyn'/'No MetSyn' to 1/0 for y_test and predictions
y_test_binary = y_test.map({'MetSyn': 1, 'No MetSyn': 0}).fillna(0).values
knn_test_preds_binary = np.where(knn_test_preds == 'MetSyn', 1, 0)
dt_test_preds_binary = np.where(dt_test_preds == 'MetSyn', 1, 0)
gnb_test_preds_binary = np.where(gnb_test_preds == 'MetSyn', 1, 0)
svm_test_preds_binary = np.where(svm_test_preds == 'MetSyn', 1, 0)
xgb_test_preds_binary = np.where(xgb_test_preds == 'MetSyn', 1, 0)
log_reg_l1_test_preds_binary = np.where(log_reg_l1_test_preds == 'MetSyn', 1, 0)
log_reg_l2_test_preds_binary = np.where(log_reg_l2_test_preds == 'MetSyn', 1, 0)
ran_for_test_preds_binary = np.where(ran_for_test_preds == 'MetSyn', 1, 0)

# Calculate the Confusion Matrices
knn_confusion_matrix = confusion_matrix(y_test_binary, knn_test_preds_binary)
decision_tree_confusion_matrix = confusion_matrix(y_test_binary, dt_test_preds_binary)
gnb_confusion_matrix = confusion_matrix(y_test_binary, gnb_test_preds_binary)
svm_confusion_matrix = confusion_matrix(y_test_binary, svm_test_preds_binary)
xgb_confusion_matrix = confusion_matrix(y_test_binary, xgb_test_preds_binary)
l1_log_reg_confusion_matrix = confusion_matrix(y_test_binary, log_reg_l1_test_preds_binary)
l2_log_reg_confusion_matrix = confusion_matrix(y_test_binary, log_reg_l2_test_preds_binary)
ran_for_confusion_matrix = confusion_matrix(y_test_binary, ran_for_test_preds_binary)

# Display the normalized Confusion Matrices
print(f'KNN Confusion Matrix:\n {knn_confusion_matrix}')
print('\n')
print(f'Decision Tree Confusion Matrix:\n {decision_tree_confusion_matrix}')
print('\n')
print(f'SVM Confusion Matrix:\n {svm_confusion_matrix}')
print('\n')
print(f'XGBoost Confusion Matrix:\n {xgb_confusion_matrix}')
print('\n')
print(f'Gaussian Naive Bayes Confusion Matrix:\n {gnb_confusion_matrix}')
print('\n')
print(f'L1 Tuned Logistics Regression Confusion Matrix:\n {l1_log_reg_confusion_matrix}')
print('\n')
print(f'L2 Tuned Logistics Regression Confusion Matrix:\n {l2_log_reg_confusion_matrix}')
print('\n')
print(f'Random Forest Confusion Matrix:\n {ran_for_confusion_matrix}')

### Confusion Matrix Normalized

# Calculate the normalized Confusion Reports.
knn_confusion_matrix = confusion_matrix(y_test, knn_test_preds, normalize = 'true')
decision_tree_confusion_matrix = confusion_matrix(y_test, knn_test_preds, normalize = 'true')
svm_confusion_matrix = confusion_matrix(y_test, knn_test_preds, normalize = 'true')
xgb_confusion_matrix = confusion_matrix(y_test, knn_test_preds, normalize = 'true')
gnb_confusion_matrix = confusion_matrix(y_test, knn_test_preds, normalize = 'true')
l1_log_reg_confusion_matrix = confusion_matrix(y_test, log_reg_l1_test_preds, normalize = 'true')
l2_log_reg_confusion_matrix = confusion_matrix(y_test, log_reg_l2_test_preds, normalize = 'true')
ran_for_confusion_matrix = confusion_matrix(y_test, ran_for_test_preds, normalize = 'true')
# Display the normalized Confusion Reports.
print(f'KNN Confusion Matrix:\n {knn_confusion_matrix}')
print('\n')
print(f'L1 Tuned Logistics Regression Confusion Matrix:\n {l1_log_reg_confusion_matrix}')
print('\n')
print(f'L2 Tuned Logistics Regression Confusion Matrix:\n {l2_log_reg_confusion_matrix}')
print('\n')
print(f'decision_tree Confusion Matrix:\n {decision_tree_confusion_matrix}')
print('\n')
print(f'gnb Confusion Matrix:\n {gnb_confusion_matrix}')
print('\n')
print(f'svm Confusion Matrix:\n {svm_confusion_matrix}')
print('\n')
print(f'xgb Confusion Matrix:\n {xgb_confusion_matrix}')
print('\n')
print(f'Random Forest Confusion Matrix:\n {ran_for_confusion_matrix}')

# Display the KNN Model confusion matrix.
ConfusionMatrixDisplay.from_estimator(best_knn, 
                                      X_test_processed, 
                                      y_test, 
                                      cmap='Blues', 
                                      normalize='true');

# Display the L1 tuned Logistics Model confusion matrix.
ConfusionMatrixDisplay.from_estimator(best_l1_log_reg_pipe, 
                                      X_test_processed, 
                                      y_test, cmap='Blues', 
                                      normalize='true');

# Display the L2 tuned Logistics Model confusion matrix.
ConfusionMatrixDisplay.from_estimator(best_l2_log_reg_pipe, 
                                      X_test_processed, 
                                      y_test, 
                                      cmap='Blues', 
                                      normalize='true')

# Display the Random Forest Model confusion matrix.
ConfusionMatrixDisplay.from_estimator(ran_for, 
                                      X_test_processed, 
                                      y_test, 
                                      cmap='Blues', 
                                      normalize='true');

# Display the Decision Tree Model confusion matrix.
ConfusionMatrixDisplay.from_estimator(decision_tree, 
                                      X_test_processed, 
                                      y_test, 
                                      cmap='Blues', 
                                      normalize='true');

# Display the Support Vector Machine Model confusion matrix.
ConfusionMatrixDisplay.from_estimator(svm, 
                                      X_test_processed, 
                                      y_test, 
                                      cmap='Blues', 
                                      normalize='true');

# Display the XGBoost Model confusion matrix.
ConfusionMatrixDisplay.from_predictions(y_test_binary, 
                                        xgb.predict(X_test_processed), 
                                        cmap='Blues', 
                                        normalize='true');

# Display the Gaussian Naive Bayes Model confusion matrix.
ConfusionMatrixDisplay.from_estimator(gnb, 
                                      X_test_processed, 
                                      y_test, 
                                      cmap='Blues', 
                                      normalize='true');
### Classification Reports

# Calculate the Classification Reports.
knn_classification_report = classification_report(y_test, knn_test_preds)
best_l1_log_reg_report = classification_report(y_test, log_reg_l1_test_preds)
best_l2_log_reg_report = classification_report(y_test, log_reg_l2_test_preds)
ran_for_classification_report = classification_report(y_test, ran_for_test_preds)
# Display the Classification Reports.
print('\n')
print(f'KNN Model Classification Report \n{knn_classification_report}');
print('\n')
print(f'Random Forest Model Classification Report \n{ran_for_classification_report}');
print('\n')
print(f'L1 Tuned Logistics Classification Report \n{best_l1_log_reg_report}');
print('\n')
print(f'L2 Tuned Logistics Classification Report \n{best_l2_log_reg_report}');

from sklearn.metrics import classification_report

# Convert true labels to numeric format
y_test_numeric = np.where(y_test == 'MetSyn', 1, 0)

# Calculate the Classification Reports.
knn_classification_report = classification_report(y_test_numeric, knn_test_preds)
best_l1_log_reg_report = classification_report(y_test_numeric, log_reg_l1_test_preds)
best_l2_log_reg_report = classification_report(y_test_numeric, log_reg_l2_test_preds)
ran_for_classification_report = classification_report(y_test_numeric, ran_for_test_preds)
svm_classification_report = classification_report(y_test_numeric, svm_test_preds)
xgb_classification_report = classification_report(y_test_numeric, xgb_test_preds)
gnb_classification_report = classification_report(y_test_numeric, gnb_test_preds)
decision_tree_classification_report = classification_report(y_test_numeric, dt_test_preds)

# Display the Classification Reports.
print('\n')
print(f'KNN Model Classification Report \n{knn_classification_report}');
print('\n')
print(f'Random Forest Model Classification Report \n{ran_for_classification_report}');
print('\n')
print(f'SVM Model Classification Report \n{svm_classification_report}');
print('\n')
print(f'XGBoost Model Classification Report \n{xgb_classification_report}');
print('\n')
print(f'Gaussian Naive Bayes Model Classification Report \n{gnb_classification_report}');
print('\n')
print(f'Decision Tree Model Classification Report \n{decision_tree_classification_report}');
print('\n')
print(f'L1 Tuned Logistics Classification Report \n{best_l1_log_reg_report}');
print('\n')
print(f'L2 Tuned Logistics Classification Report \n{best_l2_log_reg_report}');

# Calculate KNN model accuracy scores
knn_train_accuracy = knn.score(X_train_processed, y_train)
knn_test_accuracy = knn.score(X_test_processed, y_test)

# Calculate accuracy scores for the best L1 tuned Logistic Regression model
log_reg_l1_train_accuracy = best_l1_log_reg_pipe.score(X_train_processed, y_train)
log_reg_l1_test_accuracy = best_l1_log_reg_pipe.score(X_test_processed, y_test)

# Convert true labels to numeric format
y_train_numeric = np.where(y_train == 'MetSyn', 1, 0)
y_test_numeric = np.where(y_test == 'MetSyn', 1, 0)

# Calculate recall scores for the best L1 tuned Logistic Regression model
log_reg_l1_train_recall = recall_score(y_train_numeric, log_reg_l1_train_preds, pos_label=1)
log_reg_l1_test_recall = recall_score(y_test_numeric, log_reg_l1_test_preds, pos_label=1)

# Calculate precision scores for the best L1 tuned Logistic Regression model
log_reg_l1_train_precision = precision_score(y_train_numeric, log_reg_l1_train_preds, pos_label=1)
log_reg_l1_test_precision = precision_score(y_test_numeric, log_reg_l1_test_preds, pos_label=1)

# Calculate AUC scores for the best L1 tuned Logistic Regression model
log_reg_l1_train_auc = roc_auc_score(y_train, best_l1_log_reg_pipe.predict_proba(X_train_processed)[:,1])
log_reg_l1_test_auc = roc_auc_score(y_test, best_l1_log_reg_pipe.predict_proba(X_test_processed)[:,1])

# Initialize data in a dictionary series
model_summary_index = ['KNN',
                       'Logistics Regression L1 tuned', 
                               'Logistics Regression L2 tuned', 
                               'Random Forest']
d = {'Accuracy Score Train' : pd.Series([knn_train_accuracy,
                                         log_reg_l1_train_accuracy, 
                                         log_reg_l2_train_accuracy, 
                                         ran_for_train_accuracy],
                       index = model_summary_index),
     'Accuracy Score Test' : pd.Series([knn_test_accuracy,
                                        log_reg_l1_test_accuracy, 
                                        log_reg_l2_test_accuracy, 
                                        ran_for_test_accuracy],
                       index = model_summary_index),
     'Recall Score Train' : pd.Series([knn_train_recall,
                                       log_reg_l1_train_recall, 
                                       log_reg_l2_train_recall, 
                                       ran_for_train_recall],
                       index = model_summary_index),
     'Recall Score Test' : pd.Series([knn_test_recall,
                                      log_reg_l1_test_recall, 
                                      log_reg_l2_test_recall, 
                                      ran_for_test_recall],
                       index = model_summary_index),
     'Precision Score Train' : pd.Series([knn_train_precision,
                                          log_reg_l1_train_precision, 
                                          log_reg_l2_train_precision, 
                                          ran_for_train_precision],
                       index = model_summary_index),
     'Precision Score Test' : pd.Series([knn_test_precision,
                                         log_reg_l1_test_precision, 
                                         log_reg_l2_test_precision, 
                                         ran_for_test_precision],
                       index = model_summary_index),
     'AUC Score Train' : pd.Series([knn_train_auc,
                                    log_reg_l1_train_auc, 
                                    log_reg_l2_train_auc, 
                                    ran_for_train_auc],
                       index = model_summary_index),
     'AUC Score Test' : pd.Series([knn_test_auc,
                                   log_reg_l1_test_auc, 
                                   log_reg_l2_test_auc, 
                                   ran_for_test_auc],
                       index = model_summary_index),
     'F1 Macro Average' : pd.Series([.77,.81, .81, .87],
                       index = model_summary_index),
     'F1 Weighted Average' : pd.Series([.80, .83, .83, .88],
                       index = model_summary_index)}
from sklearn.metrics import f1_score

# Convert predictions to match the true labels format (1/0)
knn_test_preds_labels = np.where(knn_test_preds == 1, 1, 0)
log_reg_l1_test_preds_labels = np.where(log_reg_l1_test_preds == 1, 1, 0)
log_reg_l2_test_preds_labels = np.where(log_reg_l2_test_preds == 1, 1, 0)
ran_for_test_preds_labels = np.where(ran_for_test_preds == 1, 1, 0)
svm_test_preds_labels = np.where(svm_test_preds == 1, 1, 0)
xgb_test_preds_labels = np.where(xgb_test_preds == 1, 1, 0)
gnb_test_preds_labels = np.where(gnb_test_preds == 1, 1, 0)
dt_test_preds_labels = np.where(dt_test_preds == 1, 1, 0)

# Initialize data in a dictionary series for all models
model_summary_index = ['KNN', 'Logistics Regression L1 tuned', 'Logistics Regression L2 tuned', 'Random Forest', 'Decision Tree', 'Gaussian Naive Bayes', 'SVM', 'XGBoost']

# Ensure the previous cells are executed to define these variables
knn_train_accuracy = knn.score(X_train_processed, y_train)
knn_test_accuracy = knn.score(X_test_processed, y_test)
log_reg_l1_train_accuracy = best_l1_log_reg_pipe.score(X_train_processed, y_train)
log_reg_l1_test_accuracy = best_l1_log_reg_pipe.score(X_test_processed, y_test)
log_reg_l2_train_accuracy = best_l2_log_reg_pipe.score(X_train_processed, y_train)
log_reg_l2_test_accuracy = best_l2_log_reg_pipe.score(X_test_processed, y_test)
ran_for_train_accuracy = ran_for.score(X_train_processed, y_train)
ran_for_test_accuracy = ran_for.score(X_test_processed, y_test)
dt_train_accuracy = decision_tree.score(X_train_processed, y_train)
dt_test_accuracy = decision_tree.score(X_test_processed, y_test)
gnb_train_accuracy = gnb.score(X_train_processed, y_train)
gnb_test_accuracy = gnb.score(X_test_processed, y_test)
svm_train_accuracy = svm.score(X_train_processed, y_train)
svm_test_accuracy = svm.score(X_test_processed, y_test)
xgb_train_accuracy = xgb.score(X_train_processed, y_train)
xgb_test_accuracy = xgb.score(X_test_processed, y_test)

# Convert true labels to numeric format
y_train_numeric = np.where(y_train == 'MetSyn', 1, 0)
y_test_numeric = np.where(y_test == 'MetSyn', 1, 0)

# Calculate recall scores for the best L1 tuned Logistic Regression model
log_reg_l1_train_recall = recall_score(y_train_numeric, log_reg_l1_train_preds)
log_reg_l1_test_recall = recall_score(y_test_numeric, log_reg_l1_test_preds)
log_reg_l1_train_precision = precision_score(y_train_numeric, log_reg_l1_train_preds)
log_reg_l1_test_precision = precision_score(y_test_numeric, log_reg_l1_test_preds)

# Calculate AUC scores for the best L1 tuned Logistic Regression model
log_reg_l1_train_auc = roc_auc_score(y_train_numeric, best_l1_log_reg_pipe.predict_proba(X_train_processed)[:,1])
log_reg_l1_test_auc = roc_auc_score(y_test_numeric, best_l1_log_reg_pipe.predict_proba(X_test_processed)[:,1])

d = {
    'Accuracy Score Train': pd.Series([knn_train_accuracy, log_reg_l1_train_accuracy, log_reg_l2_train_accuracy, ran_for_train_accuracy, dt_train_accuracy, gnb_train_accuracy, svm_train_accuracy, xgb_train_accuracy], index=model_summary_index),
    'Accuracy Score Test': pd.Series([knn_test_accuracy, log_reg_l1_test_accuracy, log_reg_l2_test_accuracy, ran_for_test_accuracy, dt_test_accuracy, gnb_test_accuracy, svm_test_accuracy, xgb_test_accuracy], index=model_summary_index),
    'Recall Score Train': pd.Series([knn_train_recall, log_reg_l1_train_recall, log_reg_l2_train_recall, ran_for_train_recall, dt_train_recall, gnb_train_recall, svm_train_recall, xgb_train_recall], index=model_summary_index),
    'Recall Score Test': pd.Series([knn_test_recall, log_reg_l1_test_recall, log_reg_l2_test_recall, ran_for_test_recall, dt_test_recall, gnb_test_recall, svm_test_recall, xgb_test_recall], index=model_summary_index),
    'Precision Score Train': pd.Series([knn_train_precision, log_reg_l1_train_precision, log_reg_l2_train_precision, ran_for_train_precision, dt_train_precision, gnb_train_precision, svm_train_precision, xgb_train_precision], index=model_summary_index),
    'Precision Score Test': pd.Series([knn_test_precision, log_reg_l1_test_precision, log_reg_l2_test_precision, ran_for_test_precision, dt_test_precision, gnb_test_precision, svm_test_precision, xgb_test_precision], index=model_summary_index),
    'AUC Score Train': pd.Series([knn_train_auc, log_reg_l1_train_auc, log_reg_l2_train_auc, ran_for_train_auc, dt_train_auc, gnb_train_auc, svm_train_auc, xgb_train_auc], index=model_summary_index),
    'AUC Score Test': pd.Series([knn_test_auc, log_reg_l1_test_auc, log_reg_l2_test_auc, ran_for_test_auc, dt_test_auc, gnb_test_auc, svm_test_auc, xgb_test_auc], index=model_summary_index),
    'F1 Macro Average': pd.Series([f1_score(y_test_numeric, knn_test_preds_labels, average='macro'), f1_score(y_test_numeric, log_reg_l1_test_preds_labels, average='macro'), f1_score(y_test_numeric, log_reg_l2_test_preds_labels, average='macro'), f1_score(y_test_numeric, ran_for_test_preds_labels, average='macro'), f1_score(y_test_numeric, dt_test_preds_labels, average='macro'), f1_score(y_test_numeric, gnb_test_preds_labels, average='macro'), f1_score(y_test_numeric, svm_test_preds_labels, average='macro'), f1_score(y_test_numeric, xgb_test_preds_labels, average='macro')], index=model_summary_index),
    'F1 Weighted Average': pd.Series([f1_score(y_test_numeric, knn_test_preds_labels, average='weighted'), f1_score(y_test_numeric, log_reg_l1_test_preds_labels, average='weighted'), f1_score(y_test_numeric, log_reg_l2_test_preds_labels, average='weighted'), f1_score(y_test_numeric, ran_for_test_preds_labels, average='weighted'), f1_score(y_test_numeric, dt_test_preds_labels, average='weighted'), f1_score(y_test_numeric, gnb_test_preds_labels, average='weighted'), f1_score(y_test_numeric, svm_test_preds_labels, average='weighted'), f1_score(y_test_numeric, xgb_test_preds_labels, average='weighted')], index=model_summary_index)
}

# Convert dictionary to DataFrame
model_summary_df = pd.DataFrame(d)
print(model_summary_df)

# Display the dataframe
model_summary_df

# Plot the dataframe
model_summary_df.plot(kind='bar', figsize=(15, 8))
plt.title('Model Comparison')
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.legend(loc='best')
plt.show()

from IPython.display import display

# Display the dataframe
display(model_summary_df)

# Display line plot of scores
fig, ax = plt.subplots(nrows=1, figsize=(18,10), facecolor='w')
ax.set_facecolor('lightblue')
plt.title('Accuracy Scores', fontsize = 22, weight='bold')
sns.lineplot(data=model_summary_df['Accuracy Score Train'], color="indigo", linewidth=3, markersize=10, marker='o', label='Train');
sns.lineplot(data=model_summary_df['Accuracy Score Test'], color="magenta", linewidth=3, markersize=10, marker='o', label='Test');
plt.xlabel('Model', fontsize = 16, weight='bold')
plt.xticks(weight='bold')
ax.set_ylabel('Score', fontweight='bold', fontsize=14)
ax.set_ylim(.75, 1.05)
ax.tick_params(labelcolor='k', labelsize=8)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3);
format = StrMethodFormatter('{x:.2f}') 
ax.yaxis.set_major_formatter(format);

# Display line plot of scores
fig, ax = plt.subplots(nrows=1, figsize=(16,10), facecolor='w')
ax.set_facecolor('lightblue')
plt.title('Precision Scores', fontsize = 22, weight='bold')
sns.lineplot(data=model_summary_df['Precision Score Train'], color="indigo", linewidth=3, markersize=10, marker='o', label='Train');
sns.lineplot(data=model_summary_df['Precision Score Test'], color="magenta", linewidth=3, markersize=10, marker='o', label='Test');
plt.xlabel('Model', fontsize = 16, weight='bold')
plt.xticks(weight='bold')
ax.set_ylabel('Score', fontweight='bold', fontsize=14)
ax.set_ylim(.75, 1.05)
ax.tick_params(labelcolor='k', labelsize=8)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3);
format = StrMethodFormatter('{x:.2f}') 
ax.yaxis.set_major_formatter(format);

# Display line plot of scores
fig, ax = plt.subplots(nrows=1, figsize=(10,4), facecolor='w')
ax.set_facecolor('lightblue')
plt.title('Recall Scores', fontsize = 22, weight='bold')
sns.lineplot(data=model_summary_df['Recall Score Train'], color="indigo", linewidth=3, markersize=10, marker='o', label='Train');
sns.lineplot(data=model_summary_df['Recall Score Test'], color="magenta", linewidth=3, markersize=10, marker='o', label='Test');
plt.xlabel('Model', fontsize = 16, weight='bold')
plt.xticks(weight='bold')
ax.set_ylabel('Score', fontweight='bold', fontsize=14)
ax.set_ylim(.5, 1.05)
ax.tick_params(labelcolor='k', labelsize=8)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3);
format = StrMethodFormatter('{x:.2f}') 
ax.yaxis.set_major_formatter(format);

# Display line plot of scores
fig, ax = plt.subplots(nrows=1, figsize=(10,4), facecolor='w')
ax.set_facecolor('lightblue')
plt.title('AUC Scores', fontsize = 22, weight='bold')
sns.lineplot(data=model_summary_df['AUC Score Train'], color="indigo", linewidth=3, markersize=10, marker='o', label='Train');
sns.lineplot(data=model_summary_df['AUC Score Test'], color="magenta", linewidth=3, markersize=10, marker='o', label='Test');
plt.xlabel('Model', fontsize = 16, weight='bold')
plt.xticks(weight='bold')
ax.set_ylabel('Score', fontweight='bold', fontsize=14)
ax.set_ylim(.80, 1.05)
ax.tick_params(labelcolor='k', labelsize=8)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3);
format = StrMethodFormatter('{x:.2f}') 
ax.yaxis.set_major_formatter(format)

# Display the dataframe with model comparison
display(model_summary_df)

# Identify the best model based on the highest test accuracy score
best_model = model_summary_df['Accuracy Score Test'].idxmax()
best_model_score = model_summary_df['Accuracy Score Test'].max()

print(f'The best model to predict metabolic syndrome is: {best_model} with an accuracy score of {best_model_score:.4f}')




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
