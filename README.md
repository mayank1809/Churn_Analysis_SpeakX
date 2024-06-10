## Churn_Analysis_SpeakX
# Telecom Customer Churn Analysis Report
# Introduction
The objective of this project is to develop a predictive model that can identify customers at risk of churning for a telecommunications company. Customer churn refers to the phenomenon where customers terminate their relationship with a company, which can lead to revenue loss and decreased profitability. By predicting churn, the company can take proactive measures to retain customers and improve customer satisfaction.

# Dataset
The dataset used for this analysis is the Telco Customer Churn dataset, obtained from Kaggle. It contains information about telecom customers, including demographic data, services subscribed, and churn status.

# Approach
Data Preprocessing: The dataset was preprocessed to handle missing values, encode categorical variables, and prepare it for analysis.

Exploratory Data Analysis (EDA): 

Data Overview: 

    •	The dataset consists of customer demographics, account information, and service usage details. 

    •	Key features include tenure, MonthlyCharges, TotalCharges, gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, and PaymentMethod. 

Initial Insights: 

    •	Churn rate is around 26.5%.
    
    •	Features like tenure, MonthlyCharges, and TotalCharges show different distributions for churned and non-churned customers. 
Visualizations: 

    •	Distribution of Numerical Features: Histograms show the distribution of numerical features. 
    
    •	Churn Distribution: Count plot showing the distribution of churned vs. non-churned customers. 
    
    •	Numerical Features vs. Churn: KDE plots illustrating the relationship between numerical features and churn. 
    
    •	Correlation Matrix: Heatmap displaying correlations between features.
    
    •	Gender Distribution: Bar plot showing churn distribution by gender. 
    
    •	Payment Method Distribution: Bar plot showing churn distribution by payment method.


# Model Evaluation:

  Logistic Regression: 
  
    •	Accuracy (0.7861): Approximately 78.61% of the predictions made by the logistic regression model are correct. 
    
    •	Precision (0.6189): Out of all the customers predicted to churn, around 61.89% actually did churn. 
    
    •	Recall (0.5080): The model correctly identifies 50.80% of all actual churn cases. 
    
    •	F1-Score (0.5580): The harmonic mean of precision and recall, which balances the trade-off between the two metrics, is 55.80%. 

  Random Forest: 

    •	Accuracy (0.7875): Approximately 78.75% of the predictions made by the random forest model are correct. 
    
    •	Precision (0.6384): Out of all the customers predicted to churn, around 63.84% actually did churn. 
    
    •	Recall (0.4626): The model correctly identifies 46.26% of all actual churn cases. 
    
    •	F1-Score (0.5364): The harmonic mean of precision and recall is 53.64%. 


Confusion Matrix: The confusion matrices for both models were displayed interactively in the Streamlit app. They provide a detailed breakdown of model predictions compared to the actual class labels, aiding in the interpretation of model performance across difference classes
Check Churn Feature:

Users can input customer details through the user interface to predict churn using the trained models.

# Conclusion
Both models exhibit similar accuracy, with the random forest model slightly outperforming logistic regression in precision but slightly underperforming in recall. The choice of model may depend on whether the focus is on minimizing false positives (precision) or capturing as many actual churn cases as possible (recall). Further tuning and model evaluation might be required to optimize performance for specific business needs.

# Challenges
Imbalanced Data: Addressing imbalanced data distribution in the dataset required careful consideration during model training and evaluation.

Feature Selection: Selecting relevant features for predicting churn and engineering new features posed challenges in understanding the underlying factors influencing customer churn.

# Future Work
Model Optimization: Explore techniques for further improving the performance of churn prediction models, such as hyperparameter tuning and ensemble methods.

Customer Segmentation: Conduct more granular analysis by segmenting customers based on demographics, usage patterns, and other factors to personalize retention strategies.

Real-time Monitoring: Implement a system for real-time monitoring of customer churn using streaming data processing techniques.
