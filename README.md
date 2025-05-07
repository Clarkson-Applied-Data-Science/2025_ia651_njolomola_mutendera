# 2025_ia651_njolomola_mutendera

## Teammates : Ashwins T Njolomola and Conrad Mutendera

# Early Detection of Pulmonary Conditions Using AI Models and Patient Profile Data

This study applies a range of machine learning models including Logistic Regression, Support Vector Classifier (SVC), Decision Tree, Random Forest, and XGBoost—to structured clinical, demographic, and lifestyle data in order to predict pulmonary disease. The analysis includes feature engineering, scaling of continuous variables, and hyperparameter tuning using GridSearchCV. Model performance is evaluated using F1-score and ROC-AUC. Among the classifiers, XGBoost demonstrated the best predictive performance, outperforming other models in accuracy and F1-score. To ensure transparency and trust in predictions, the study incorporates interpretability techniques such as feature importance, permutation importance, and LIME explanations.

## PROJECT OVERVIEW

This project emphasizes model interpretability and performance transparency. Techniques such as LIME, feature importance plots, and permutation importance provide insights into the driving factors behind each prediction. Visual diagnostics including confusion matrices, ROC curves, and comparison charts were employed to evaluate and communicate each model’s classification behavior. Furthermore, exploratory analysis through PCA adds dimensional insight, supporting the interpretability of complex patterns within the dataset.

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

## Process Overview

To improve the predictive modeling technique, an iterative approach was used throughout the project.  Early tests using baseline models, like decision trees and logistic regression, revealed data trends but had accuracy and generalizability issues.  Additional measures to enhance model performance included trying a larger variety of techniques, such as Random Forest and XGBoost, feature scaling, and hyperparameter tuning with GridSearchCV.  Overfitting on the training data and the necessity of changing the models that were initially chosen due to subpar validation results were among the difficulties faced.  In the end, a reliable and understandable model for predicting pulmonary diseases was chosen after each cycle offered more insight into the dataset.

## Exploratory Data Analysis (EDA)

. X Variables (Features): AGE, GENDER, SMOKING, ALCOHOL_CONSUMPTION, EXPOSURE_TO_POLLUTION, BREATHING_ISSUE, THROAT_DISCOMFORT, ENERGY_LEVEL, OXYGEN_SATURATION

. Y Variable (Target): PULMONARY_DISEASE

. Problem Type: Binary Classification

. Number of Observations: 5,000 patients

. Number of Features: 18 (after preprocessing and encoding)

## Features Dataset

![image](https://github.com/user-attachments/assets/f25dd466-3129-4502-b744-4966a8bc975f)


# Target Variable

. The target variable in this project is PULMONARY_DISEASE, a binary classification label that indicates whether or not a patient shows clinical signs of pulmonary disease. It consists of two classes encoded as:

. 0 – No Pulmonary Disease: The patient exhibits no clinical or lifestyle indicators suggesting the presence of lung disease.

. 1 – Pulmonary Disease: The patient presents with symptoms or risk factors consistent with respiratory or pulmonary impairment, such as breathing difficulties, low oxygen saturation, or exposure to environmental risk factors.

. Pulmonary diseases include a wide range of conditions that impair lung function, such as chronic bronchitis, asthma, and other respiratory disorders. Early prediction of such conditions based on observable patient characteristics can support clinical decisions, promote early intervention, and ultimately improve patient outcomes.

## Feature Distrubution

![image](https://github.com/user-attachments/assets/ccc4a350-0e0d-4ec3-b6a7-598ea74d4238)


![image](https://github.com/user-attachments/assets/a0399e23-8257-401f-9275-6488eb9412ab)


![image](https://github.com/user-attachments/assets/c24269f2-0d12-4b31-a380-e28038c9dc79)


![image](https://github.com/user-attachments/assets/5343d141-bba2-4013-a16a-4c5a84eb098a)


## Correlation Analysis


![image](https://github.com/user-attachments/assets/9fcd13ba-845a-474a-b9cb-f208accaae95)

The correlation heatmap revealed generally weak linear relationships among most numerical features in the dataset. While some variables such as OXYGEN_SATURATION and ENERGY_LEVEL showed mild positive correlations, the majority of features exhibited minimal interdependence. AGE had a slight negative correlation with ENERGY_LEVEL, and BREATHING_ISSUE showed weak associations with several other clinical symptoms. Importantly, no strong multicollinearity was observed between predictors, indicating that each feature may contribute uniquely to the model. As a result, all features were retained for further modeling to capture both linear and potential nonlinear interactions.

## Train/Test Split

To ensure robust model evaluation and prevent overfitting, the dataset was split into three subsets: 70% for training, 15% for validation, and 15% for testing. The training set was used to fit the models, while the validation set supported model comparison and hyperparameter tuning. The final test set was held out entirely to assess the model's generalization on unseen data. A stratified splitting strategy was applied to preserve the class distribution of the PULMONARY_DISEASE target variable across all sets, maintaining a balanced representation of both positive and negative cases
## Feature Engineering

- To prepare the dataset for modeling and enhance predictive performance, the following feature engineering steps were applied:

- Label Encoding: Binary categorical variables such as GENDER, SMOKING, ALCOHOL_CONSUMPTION, and EXPOSURE_TO_POLLUTION were encoded into numerical format (0 and 1).

- Feature Scaling: Continuous variables (AGE, ENERGY_LEVEL, and OXYGEN_SATURATION) were standardized using StandardScaler to normalize their ranges and improve model performance.

- Target Encoding: The target variable PULMONARY_DISEASE was label-encoded as a binary class (0 = No, 1 = Yes) for classification tasks.

- Correlation Review: A correlation heatmap was used to evaluate multicollinearity. As no strong correlations were observed, all features were retained for modeling.

## Principal Component Analysis (PCA)

Principal Component Analysis (PCA) was performed to explore the structure of the dataset and visualize how much variance is captured by each component. PCA was applied specifically to the standardized continuous features: AGE, ENERGY_LEVEL, and OXYGEN_SATURATION. The analysis was not intended for dimensionality reduction in modeling, but rather to gain insight into feature variance and potential class separation. The resulting scree plot (shown below) illustrates the proportion of variance explained by each principal component. The first two components captured the majority of the variance and were used to generate a 2D projection, revealing partial separation between patients with and without pulmonary disease. However, all features were retained in the final models to preserve information, especially from non-linear relationships not captured by PCA alone.

![image](https://github.com/user-attachments/assets/32ef62fe-85a3-4e99-b732-8c5bcd7e97f8)


![image](https://github.com/user-attachments/assets/942a44bb-f0b0-4395-9d6d-74fcf69e1761)

![image](https://github.com/user-attachments/assets/14aa6a83-c323-4632-a2db-3950012e7aee)

The PCA results revealed that a substantial portion of the dataset's variance was captured by the first few principal components. Specifically, the top components accounted for the majority of variability across the standardized continuous features (AGE, ENERGY_LEVEL, OXYGEN_SATURATION). While dimensionality reduction was not applied to the final model, PCA helped uncover the underlying structure of the data and provided visual insights into class separation. This exploratory step reinforced the decision to retain all original features for modeling, ensuring that both linear and non-linear patterns contributing to pulmonary disease prediction were preserved.

# Model Fitting
## Model Selection

Multiple supervised machine learning algorithms were evaluated to identify the best model for predicting pulmonary disease. The models included 

. Logistic Regression

. Support Vector Classifier (SVC)

. Decision Tree

. Random Forest

. XGBoost

Each model was trained on the same preprocessed dataset and evaluated using consistent performance metrics. 

## Hyperparameter Tuning And

Hyperparameter tuning was performed using GridSearchCV with 5-fold stratified cross-validation to ensure reliable model comparison, prevent overfitting, and identify the optimal parameter configurations for each algorithm. 

## Validation and Metrics
Performance was assessed based on:

. Accuracy

. Precision

. Recall

. F1-score

. ROC-AUC

## Consolidated Model Metrics

![image](https://github.com/user-attachments/assets/d383e3cd-970b-4685-99e0-81bce636ea0e)


. Here are the confusion matrices and ROC curves for the five evaluated models: Logistic Regression, Support Vector Classifier (SVC), Decision Tree, Random Forest, and XGBoost. These visualizations help assess each model’s classification performance, including their ability to distinguish between patients with and without pulmonary disease.

![image](https://github.com/user-attachments/assets/55c9053b-51ed-4b24-a26f-81d2b9d2c782)



![image](https://github.com/user-attachments/assets/90347634-632e-4394-9b70-aba8b88bb243)



![image](https://github.com/user-attachments/assets/ed450da3-f418-47cb-86ef-43e87dce0970)



![image](https://github.com/user-attachments/assets/4931f486-c6a0-4e7a-9443-e9de1b42470f)


![image](https://github.com/user-attachments/assets/6772b758-cd70-4f39-8c3e-04fc3c232546)



# Feature Importance

![image](https://github.com/user-attachments/assets/c746ed6c-4bc7-49d2-8f1d-972ba1ee2e16)

![image](https://github.com/user-attachments/assets/30c30578-4524-45f6-8f41-fe87a27464bf)


## Feature Contribution Table


![image](https://github.com/user-attachments/assets/a624e237-7039-4be8-913a-b10ce29daddb)


<img width="707" alt="final summary" src="https://github.com/user-attachments/assets/17a96310-ab69-4dc4-84a9-469aebf73808" />


# Future Work
While this study presents promising results in predicting pulmonary disease using machine learning, several avenues exist to enhance and extend its impact:

Incorporate More Clinical Variables
Future iterations could integrate more granular medical data such as spirometry results, radiological findings, or blood biomarkers to boost predictive accuracy and model robustness.

Model Generalization and Validation
External validation with real-world clinical datasets from different hospitals or geographic regions is crucial to assess generalizability and reduce dataset-specific bias.

Real-Time and Longitudinal Data
Leveraging wearable health monitors or IoT devices for continuous tracking of symptoms like oxygen saturation or breathing patterns could enable dynamic risk prediction over time.

Model Deployment in Clinical Settings
The final model can be deployed via a web-based dashboard (using tools like Streamlit or Flask) for use by medical practitioners. Integration with Electronic Health Records (EHR) systems could streamline workflows.

Patient-Centered Interfaces
Developing mobile applications that provide patients with risk feedback and lifestyle recommendations based on model predictions could empower proactive health management.

Explainability Enhancements
Incorporating more robust model interpretability tools such as SHAP values can further enhance trust and adoption in clinical environments, especially for black-box models like XGBoost.

Ethical and Fairness Audits
Future work should evaluate model performance across demographic subgroups to ensure fairness and mitigate potential bias, especially regarding age, gender, or socioeconomic status.

# Conclusion

This project demonstrates the transformative potential of machine learning in predicting pulmonary disease with high accuracy and clinical relevance. By integrating structured data on patient demographics, clinical symptoms, and lifestyle behaviors, the models provide a non-invasive, scalable approach for early disease detection. Among the models tested, XGBoost emerged as the most effective, outperforming others in terms of F1-score and overall predictive performance, while maintaining strong interpretability through feature importance and LIME explanations.

Beyond technical achievement, this work lays the foundation for deployable, explainable AI tools in healthcare—tools that can assist clinicians in triaging patients, improving diagnostic speed, and optimizing treatment planning. As the demand for precision medicine grows, such predictive systems have the potential to reduce hospital burden, increase patient survival rates, and drive forward a data-informed future in medical care. With further validation and integration into clinical systems, the model developed here could meaningfully impact public health outcomes, especially in under-resourced or high-risk populations.


































