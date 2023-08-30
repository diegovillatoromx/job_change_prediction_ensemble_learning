# Data Science Job Change Prediction Model using Ensemble Learning
In the realm of Big Data and Data Science, a company specializes in recruiting data scientists from those who successfully complete their training courses. With a large pool of enrolled individuals, the company aims to differentiate candidates who genuinely intend to join their workforce post-training, from those who are actively seeking new job opportunities. This distinction holds the key to reducing costs, enhancing training quality, and optimizing course planning. Leveraging demographic, educational, and experiential data gathered during candidate enrollment, the task at hand is to develop predictive models that ascertain the likelihood of a candidate either seeking alternative employment or committing to the company. This analysis not only informs strategic human resource decisions, but also provides insights into the factors influencing employee decisions concerning their future career paths. 
 
## Table of Contents

- [Description](#description)
- [Architecture](#architecture)
- [Features](#features)
- [Modular_Code_Overview](#modular_code_overview)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#contact)

## Description

This project is designed to understand the factors that lead a person to leave their current job for HR research too. By model(s) that uses the current credentials,demographics,experience data you will predict the probability of a candidate to look for a new job or will work for the company, as well as interpreting affected factors on employee decision.

In the realm of Big Data and Data Science, a company specializes in recruiting data scientists from those who successfully complete their training courses. With a large pool of enrolled individuals, the company aims to differentiate candidates who genuinely intend to join their workforce post-training, from those who are actively seeking new job opportunities. This distinction holds the key to reducing costs, enhancing training quality, and optimizing course planning. Leveraging demographic, educational, and experiential data gathered during candidate enrollment, the task at hand is to develop predictive models that ascertain the likelihood of a candidate either seeking alternative employment or committing to the company. This analysis not only informs strategic human resource decisions, but also provides insights into the factors influencing employee decisions concerning their future career paths.

We've previously examined the functionality of the logistic regression model using this dataset in the initial project of this series: [Predictive Employee Intent Analysis: Identifying Future Job Seekers and Company Devotees using Demographics and Experience Data](https://github.com/diegovillatoromx/Strategic_Workforce_Analysis_Predicting_Job_Transition).

Additionally, we've implemented the decision tree algorithm in our second project: [Data Science Job Change Prediction Model using Decision Trees](https://github.com/diegovillatoromx/Job_change_prediction_decision_trees).
It's advisable to review these two projects beforehand as we delve into ensemble techniques.

## Architecture

![diagram](https://github.com/diegovillatoromx/job_change_prediction_ensemble_learning/blob/main/architecture_diagrama_ensemble_learning.png)
## Data Description
The CSV consists of around 19,158 rows and 14 columns in the [dataset](https://github.com/diegovillatoromx/job_change_prediction_ensemble_learning/blob/main/input/DS_Job_Change_Data.csv)
#### Features:
- enrollee_id : Unique ID for candidate
- city: City code
- city_ development _index : Developement index of the city (scaled)
- gender: Gender of candidate
- relevent_experience: Relevant experience of candidate
- enrolled_university: Type of University course enrolled if any
- education_level: Education level of candidate
- major_discipline :Education major discipline of candidate
- experience: Candidate total experience in years
- company_size: No of employees in current employer's company
- company_type : Type of current employer
- last_new_job: Difference in years between previous job and current job
- training_hours: training hours completed
- target: 0 – Not looking for job change, 1 – Looking for a job change

## Modular_Code_Overview

```
  input
    |_DS_Job_Change_Data.csv

  ML_pipeline
    |_evaluate_metrics.py
    |_lime.py
    |_ml_model.py
    |_utils.py

  Tutorial
    |_decision_tree.ipynb

  output
    |_LIME_reports folder
    |_models folder
    |_ROC_curves folder
```
1. Input - It contains all the data that we have for analysis. There is one csv
file in our case:
   - DS_Job_Change_Data.csv
2. ML_Pipeline
   - The ML_pipeline is a folder that contains all the functions put into different
      python files, which are appropriately named. These python functions are
      then called inside the engine.py file.

3. Output
   – Output folder – The output folder contains three subfolders.
     - LIME_reports - contains the LIME reports generated for all three algorithms.
     - Models - contains the models generated for all three algorithms.
     - ROC_curves - contains the ROC curves generated for all three algorithms.
4. Tutorial - This is a reference folder. It contains the notebook tutorial.

## Usage

How to utilize and operate the Data Science project after completing the installation steps.
### Data Preparation
Before analysis, prepare data by loading and processing it:
1. ##### Import the required libraries
    ```terminal
    import pickle
    from ML_Pipeline.utils import read_data,inspection,null_values
    from ML_Pipeline.ml_model import prepare_model_smote,run_model
    from ML_Pipeline.evaluate_metrics import confusion_matrix,roc_curve
    from ML_Pipeline.lime import lime_explanation
    import matplotlib.pyplot as plt
    ```
2. ##### Data loading
    If data is in CSV format, load it using Pandas:
    ```terminal
    datapath = 'input/data_regression.csv'
    df = read_data(datapath)
    df.head(5)
    ```
    ![df_head](https://github.com/diegovillatoromx/ensemble_learning/blob/main/images/dfhead.png)
 
3. #### Inspection and cleaning the data
    ```terminal
    x = inspection(df)
    ```
    ![inspection](https://github.com/diegovillatoromx/ensemble_learning/blob/main/images/inspection.png)
4. #### Cleaning and Preprocessing:
   Clean data by handling missing values, normalization, etc.
    ```terminal
    column_names = df.columns.tolist()
    target = column_names[-1] 
    cols_to_exclude = column_names[0:4] 
    df = null_values(df)
    ```
### Training Model
Perform analysis and modeling on prepared data:

1. #### Model Selection
   Selecting only the numerical columns and excluding the columns we specified in the function
   ```terminal
    X_train, X_test, y_train, y_test = prepare_model_smote(df,class_col='churn',
                                                 cols_to_exclude=['customer_id','phone_no', 'year'])
    ```
### Evaluation

1. #### Evaluation Metrics
   ```terminal
   model_rf,y_pred = run_model('random',X_train,X_test,y_train,y_test)
   ```
   ![running_model](https://github.com/diegovillatoromx/ensemble_learning/blob/main/images/run_model.png)


2. #### Performance metrics
   ```terminal
   conf_matrix = confusion_matrix(y_test,y_pred)
   ```
   ![running_model](https://github.com/diegovillatoromx/ensemble_learning/blob/main/images/cof_matrix.png)

   ```terminal
   import os
   os.makedirs("output/ROC_curves", exist_ok=True)
   roc_val = roc_curve(model_rf, X_test, y_test) # plot the roc curve
   plt.savefig("output/ROC_curves/ROC_Curve_rf.png") # plot the featu
   ```
   ![ROC](https://github.com/diegovillatoromx/ensemble_learning/blob/main/images/ROC_Curve_rf.png)

   ```terminal
   os.makedirs("output/models", exist_ok=True)
   pickle.dump(model_rf, open('output/models/model_rf.pkl', 'wb'))
   ```
   
3. #### Feature Importance
   ```terminal
   os.makedirs("output/LIME_reports", exist_ok=True)
   lime_exp = lime_explanation(model_rf,X_train,X_test,['Not Churn','Churn'],1)
   lime_exp.savefig('output/LIME_reports/lime_report_rf.jpg')
   ```
   ![running_model](https://github.com/diegovillatoromx/ensemble_learning/blob/main/images/lime_report_rf.jpg)


## Contribution Guidelines
  1. Focus changes on specific improvements.
  2. Follow project's coding style.
  3. Provide detailed descriptions in pull requests.
## Reporting Issues
  Use "Issues" to report bugs or suggest improvements.
# Contact
For questions or contact, [Mail](diegovillatormx@gmail.com) or [Twitter](https://twitter.com/diegovillatomx) .

