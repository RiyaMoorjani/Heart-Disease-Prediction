# Heart-Disease-Prediction
A machine learning pipeline for predicting heart disease using the Z-Alizadeh Sani Heart Disease Dataset, implementing and comparing 7 classification algorithms with an interactive Streamlit web interface.

#Overview
This project builds a complete ML pipeline that:

Preprocesses and normalizes medical patient data
Selects the most predictive features using Correlation-Based Feature Selection (CFS)
Trains and compares 7 machine learning classifiers
Evaluates models using standard classification metrics
Serves predictions through an interactive Streamlit web app


#Methodology
1. Data Preprocessing

Missing values filled with median (numerical) and mode (categorical)
Categorical features encoded with Label Encoding
All features normalized to [0, 1] using Min-Max Scaling
Data split: 80% training / 20% testing (stratified)

2. Feature Selection
Method: Correlation-Based Feature Subset Selection (CFS) with Best First Search heuristic using Random Forest importance scores.

Original features: 54
Selected features: 30
Reduction: ~44%

3. Models Trained
#Model
1.Logistic Regression
2.Naïve Bayes
3.K-Nearest Neighbor (K=5)
4.Support Vector Machine (RBF kernel)
5.Decision Tree
6.Random Forest (100 estimators)
7.Multilayer Perceptron (100→50 neurons)

4. Evaluation Metrics

Accuracy
Precision
Recall
F1-Score
ROC-AUC
Confusion Matrix


5. Evaluation Metrics

Accuracy
Precision
Recall
F1-Score
ROC-AUC
Confusion Matrix

6. Key Findings

Feature selection (30 features) achieved comparable or better performance than using all 54 features
Random Forest consistently performed best across all metrics
Top predictive features aligned with known cardiac risk factors (chest pain type, age, ECG results)

Results
Results are generated after running main.py and saved to outputs/model_results_complete.csv.

