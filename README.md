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

#Principal Component Analysis (PCA)

Principal Component Analysis (PCA) was performed to explore the structure of the dataset and visualize how much variance is captured by each component. PCA was applied specifically to the standardized continuous features: AGE, ENERGY_LEVEL, and OXYGEN_SATURATION. The analysis was not intended for dimensionality reduction in modeling, but rather to gain insight into feature variance and potential class separation. The resulting scree plot (shown below) illustrates the proportion of variance explained by each principal component. The first two components captured the majority of the variance and were used to generate a 2D projection, revealing partial separation between patients with and without pulmonary disease. However, all features were retained in the final models to preserve information, especially from non-linear relationships not captured by PCA alone.

![image](https://github.com/user-attachments/assets/39f7ce2a-d33f-4e26-a63a-6e9236c83ac2)
![image](https://github.com/user-attachments/assets/2d52f8ff-764e-4749-9154-3f0e994d2b01)
![image](https://github.com/user-attachments/assets/7cdb589e-1053-4830-80d4-d8c5084e6549)

The PCA results revealed that a substantial portion of the dataset's variance was captured by the first few principal components. Specifically, the top components accounted for the majority of variability across the standardized continuous features (AGE, ENERGY_LEVEL, OXYGEN_SATURATION). While dimensionality reduction was not applied to the final model, PCA helped uncover the underlying structure of the data and provided visual insights into class separation. This exploratory step reinforced the decision to retain all original features for modeling, ensuring that both linear and non-linear patterns contributing to pulmonary disease prediction were preserved.

## Model Fitting
# Train/Test Split

To ensure robust model evaluation and prevent overfitting, the dataset was split into three subsets: 70% for training, 15% for validation, and 15% for testing. The training set was used to fit the models, while the validation set supported model comparison and hyperparameter tuning. The final test set was held out entirely to assess the model's generalization on unseen data. A stratified splitting strategy was applied to preserve the class distribution of the PULMONARY_DISEASE target variable across all sets, maintaining a balanced representation of both positive and negative cases

# Model Selection

Multiple supervised machine learning algorithms were evaluated to identify the best model for predicting pulmonary disease. The models included 
. Logistic Regression
. Support Vector Classifier (SVC)
. Decision Tree
. Random Forest
. XGBoost
Each model was trained on the same preprocessed dataset and evaluated using consistent performance metrics. 

# Hyperparameter Tuning And

Hyperparameter tuning was performed using GridSearchCV with 5-fold stratified cross-validation to ensure reliable model comparison, prevent overfitting, and identify the optimal parameter configurations for each algorithm. 

# Validation and Metrics
Performance was assessed based on:
.Accuracy
.Precision
.Recall
.F1-score
.ROC-AUC
# Consolidated Model Metrics
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.84	0.81	0.83	0.82	0.89
SVC	0.85	0.82	0.84	0.83	0.9
Decision Tree	0.83	0.8	0.82	0.81	0.86
Random Forest	0.89	0.87	0.89	0.88	0.93
XGBoost	0.88	0.86	0.87	0.86	0.92
![image](https://github.com/user-attachments/assets/e833e9c5-cee4-4827-b413-af05fe9c78ce)























