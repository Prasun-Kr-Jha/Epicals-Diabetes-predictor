Epicals – Diabetes Risk Predictor

Epicals – Diabetes Risk Predictor is a machine learning based application designed to predict the risk of diabetes using key medical parameters. The project focuses on early risk identification using supervised learning techniques and demonstrates the practical application of machine learning in healthcare.

Project Objective

The main objective of this project is to analyze patient health data and predict whether a person is likely to have diabetes. This helps in understanding how machine learning models can assist in early-stage medical risk assessment.

Features

• Predicts diabetes risk based on medical inputs
• Uses machine learning classification algorithms
• Compares Logistic Regression and Random Forest models
• Includes model evaluation using accuracy, confusion matrix, and ROC curve
• Uses feature scaling for improved performance
• Provides real-time predictions using a deployed Streamlit application

Machine Learning Models Used

Logistic Regression

Random Forest Classifier

Random Forest is selected as the final model due to higher accuracy and better feature importance interpretation.

Dataset Description

The dataset consists of medical attributes commonly used for diabetes diagnosis:

• Pregnancies
• Glucose Level
• Blood Pressure
• Skin Thickness
• Insulin
• Body Mass Index (BMI)
• Diabetes Pedigree Function
• Age

Target Variable:
0 – Non-Diabetic
1 – Diabetic

Project Structure

app.py – Main Streamlit application
train_diabetes.py – Model training script
check_accuracy.py – Model evaluation script
diabetes.csv – Dataset
final_rf_model.joblib – Final trained model
logreg_model.joblib – Logistic Regression model
scaler.joblib – Feature scaler
feature_importances.csv – Feature importance values
confusion_matrix.png – Confusion matrix
roc_curve.png – ROC curve
requirements.txt – Python dependencies

How to Run the Project Locally

Clone the repository

Install required dependencies using requirements.txt

Run the application using Streamlit

Command:

streamlit run app.py

Model Evaluation

The model is evaluated using:

• Accuracy score
• Confusion Matrix
• ROC Curve

These metrics help measure classification performance and reliability.

Deployment

The application is deployed using Streamlit Cloud. The deployment allows real-time prediction through a web-based interface accessible via a public URL.

Use Cases

• Academic mini or major project
• Hackathon project
• Healthcare data analysis practice
• Machine learning deployment learning

Disclaimer

This application is intended for educational purposes only. It does not replace professional medical diagnosis or treatment.

Author:-
Prasun Kumar Jha


Future Scope

• Add probability-based risk levels
• Improve UI design
• Add explainable AI features
• Integrate cloud database


