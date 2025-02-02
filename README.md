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
