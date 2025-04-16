# 2025_ia651_njolomola_mutendera
Code for 2025 IA651 Final Project. Teammates Ashwins T Njolomola and Conrad Mutendera

This study uses a variety of machine learning models to analyze clinical, demographic, and lifestyle data in order to predict the development of pulmonary disease.  Logistic Regression, Support Vector Classifier (SVC), Decision Tree, Random Forest, and XGBoost are among the classifiers that are compared in this investigation, which also includes feature scaling and model training on structured data.  F1-score and ROC-AUC are used to determine the final model selection, and the Random Forest classifier performs better than the others.  To guarantee clear forecasts, feature importance, permutation importance, and LIME explanations are used to address model interpretability.
## PROJECT OVERVIEW
This project's objective is to develop and assess machine learning models that use clinical, lifestyle, and demographic characteristics to forecast the risk of pulmonary illness.  Model training with classifiers like Logistic Regression, Support Vector Classifier (SVC), Decision Tree, Random Forest, and XGBoost is part of the analysis, along with data preparation and feature scaling.  Accuracy, precision, recall, F1-score, and ROC-AUC are among the metrics used to evaluate the models' performance using stratified K-fold cross-validation.  Utilizing LIME, feature importance, and permutation importance, interpretability is addressed.  Visualizations such confusion matrices, ROC curves, and bar charts showing model performance are used to display the results.
## DATASET
The dataset used for this project is focused on predicting the presence of pulmonary disease using a combination of demographic, clinical, and lifestyle-related features. The dataset includes 5,000 patient records and was sourced from a publicly available lung health study. Each record contains attributes that reflect the patient's age, gender, physiological measurements, and behavioral factors. 
## Key fields include:

AGE: Patient’s age

GENDER: Biological sex of the patient

SMOKING: Whether the patient is a smoker

ALCOHOL_CONSUMPTION: Frequency of alcohol intake

EXPOSURE_TO_POLLUTION: Exposure level to air pollution

BREATHING_ISSUE: Presence of breathing difficulties

THROAT_DISCOMFORT: Experience of throat irritation

OXYGEN_SATURATION: Measured oxygen level in the blood

ENERGY_LEVEL: Self-reported energy levels

PULMONARY_DISEASE: Target variable indicating the presence of disease (Yes/No)

This dataset is valuable for predicting the likelihood of pulmonary disease based on lifestyle and clinical symptoms, enabling early screening and support for medical diagnosis.

## Prediction Objective

This project's primary objective is to predict a patient's likelihood of having pulmonary (lung) disease using a combination of clinical, demographic, and lifestyle characteristics. By leveraging machine learning models, the system aims to support therapeutic decision-making by enabling early identification of individuals at risk. Such early detection can significantly improve patient outcomes by facilitating timely interventions and guiding effective treatment planning.

# Process Overview

## Process Overview
To improve the predictive modeling technique, an iterative approach was used throughout the project.  Early tests using baseline models, like decision trees and logistic regression, revealed data trends but had accuracy and generalizability issues.  Additional measures to enhance model performance included trying a larger variety of techniques, such as Random Forest and XGBoost, feature scaling, and hyperparameter tuning with GridSearchCV.  Overfitting on the training data and the necessity of changing the models that were initially chosen due to subpar validation results were among the difficulties faced.  In the end, a reliable and understandable model for predicting pulmonary diseases was chosen after each cycle offered more insight into the dataset.

## Exploratory Data Analysis (EDA)
X Variables (Features): AGE, GENDER, SMOKING, ALCOHOL_CONSUMPTION, EXPOSURE_TO_POLLUTION, BREATHING_ISSUE, THROAT_DISCOMFORT, ENERGY_LEVEL, OXYGEN_SATURATION
Y Variable (Target): PULMONARY_DISEASE
Problem Type: Binary Classification
Number of Observations: 5,000 patients
Number of Features: 18 (after preprocessing and encoding)
## Features Dataset
<img width="432" alt="image" src="https://github.com/user-attachments/assets/fd18328c-8f8e-4a4b-88d4-f7f20bb3edf1" />

# Target Variable

The target variable in this project is PULMONARY_DISEASE, a binary classification label that indicates whether or not a patient shows clinical signs of pulmonary disease. It consists of two classes encoded as:

0 – No Pulmonary Disease: The patient exhibits no clinical or lifestyle indicators suggesting the presence of lung disease.

1 – Pulmonary Disease: The patient presents with symptoms or risk factors consistent with respiratory or pulmonary impairment, such as breathing difficulties, low oxygen saturation, or exposure to environmental risk factors.

Pulmonary diseases include a wide range of conditions that impair lung function, such as chronic bronchitis, asthma, and other respiratory disorders. Early prediction of such conditions based on observable patient characteristics can support clinical decisions, promote early intervention, and ultimately improve patient outcomes.

# Feature Distrubution
![image](https://github.com/user-attachments/assets/ccc4a350-0e0d-4ec3-b6a7-598ea74d4238)


![image](https://github.com/user-attachments/assets/c04fa05e-3e54-4785-b16d-ab018cba6607)


![image](https://github.com/user-attachments/assets/a0399e23-8257-401f-9275-6488eb9412ab)


![image](https://github.com/user-attachments/assets/c24269f2-0d12-4b31-a380-e28038c9dc79)


![image](https://github.com/user-attachments/assets/5343d141-bba2-4013-a16a-4c5a84eb098a)


# Correlation Analysis


![image](https://github.com/user-attachments/assets/9fcd13ba-845a-474a-b9cb-f208accaae95)

The correlation heatmap revealed generally weak linear relationships among most numerical features in the dataset. While some variables such as OXYGEN_SATURATION and ENERGY_LEVEL showed mild positive correlations, the majority of features exhibited minimal interdependence. AGE had a slight negative correlation with ENERGY_LEVEL, and BREATHING_ISSUE showed weak associations with several other clinical symptoms. Importantly, no strong multicollinearity was observed between predictors, indicating that each feature may contribute uniquely to the model. As a result, all features were retained for further modeling to capture both linear and potential nonlinear interactions.

# Feature Engineering

To prepare the dataset for modeling and enhance predictive performance, the following feature engineering steps were applied:

Label Encoding: Binary categorical variables such as GENDER, SMOKING, ALCOHOL_CONSUMPTION, and EXPOSURE_TO_POLLUTION were encoded into numerical format (0 and 1).

Feature Scaling: Continuous variables (AGE, ENERGY_LEVEL, and OXYGEN_SATURATION) were standardized using StandardScaler to normalize their ranges and improve model performance.

Target Encoding: The target variable PULMONARY_DISEASE was label-encoded as a binary class (0 = No, 1 = Yes) for classification tasks.

Correlation Review: A correlation heatmap was used to evaluate multicollinearity. As no strong correlations were observed, all features were retained for modeling.



















