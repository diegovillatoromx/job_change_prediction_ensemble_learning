# Data Science Job Change Prediction Model using Ensemble Learning
In the realm of Big Data and Data Science, a company specializes in recruiting data scientists from those who successfully complete their training courses. With a large pool of enrolled individuals, the company aims to differentiate candidates who genuinely intend to join their workforce post-training, from those who are actively seeking new job opportunities. This distinction holds the key to reducing costs, enhancing training quality, and optimizing course planning. Leveraging demographic, educational, and experiential data gathered during candidate enrollment, the task at hand is to develop predictive models that ascertain the likelihood of a candidate either seeking alternative employment or committing to the company. This analysis not only informs strategic human resource decisions, but also provides insights into the factors influencing employee decisions concerning their future career paths. 
 
## Table of Contents

- [Description](#description)
- [Architecture](#architecture)
- [Features](#features)
- [Modular_Code_Overview](#modular_code_overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#contact)

## Description

Our case study focuses on a churn dataset, where "churned customers" are those ending relationships with their current company. XYZ, a service provider, offers a one-year subscription plan and wants to predict customer  renewal.

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
