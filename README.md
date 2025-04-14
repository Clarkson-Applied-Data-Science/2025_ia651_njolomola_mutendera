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
## Features





