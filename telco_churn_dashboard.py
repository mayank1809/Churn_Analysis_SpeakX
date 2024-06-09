import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handling missing values
df.dropna(inplace=True)  # Drop rows with missing values

# Convert TotalCharges to numeric, setting errors='coerce' will convert non-convertible values to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop any remaining rows with NaN values
df.dropna(inplace=True)

# List of categorical columns
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

# Manually map categorical columns to numeric values
mappings = {
    'gender': {'Female': 0, 'Male': 1},
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'PhoneService': {'No': 0, 'Yes': 1},
    'MultipleLines': {'No': 0, 'Yes': 1, 'No phone service': 2},
    'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
    'OnlineSecurity': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'OnlineBackup': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'DeviceProtection': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'TechSupport': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'StreamingTV': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'StreamingMovies': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'No': 0, 'Yes': 1},
    'PaymentMethod': {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}
}

for col in categorical_columns:
    df[col] = df[col].map(mappings[col])

# Convert the target variable 'Churn' to numeric
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Feature Engineering: Creating a new feature 'SeniorCitizen_Partner'
df['SeniorCitizen_Partner'] = df['SeniorCitizen'] * df['Partner']

# Split the data into training and testing sets
X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical features
numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Train a logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

# Train a random forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluation function
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, precision, recall, f1, cm

# Streamlit app
st.title("Telecom Customer Churn Analysis")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select Analysis", ["Data Overview", "EDA", "Model Evaluation", "Check Churn"])

# Data Overview
if option == "Data Overview":
    st.header("Data Overview")
    st.write("First 5 rows of the dataset:")
    st.write(df.head())
    st.write("Summary statistics of the dataset:")
    st.write(df.describe())

# Exploratory Data Analysis (EDA)
elif option == "EDA":
    st.header("Exploratory Data Analysis")
    eda_option = st.radio("Select EDA", ["Distribution of Numerical Features", "Churn Distribution", "Numerical Features vs Churn", "Correlation Matrix", "Gender Distribution","Payment Method Distribution"])
    
    if eda_option == "Distribution of Numerical Features":
        for col in numerical_columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True, color='darkblue')
            plt.title(f'Distribution of {col}')
            st.pyplot(plt)
            st.write(df[col].describe())  # Summary statistics
            
    elif eda_option == "Churn Distribution":
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Churn', data=df, color='darkblue')
        plt.title('Distribution of Churn')
        st.pyplot(plt)
        st.write(df['Churn'].value_counts())  # Summary statistics
        
    elif eda_option == "Numerical Features vs Churn":
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
        for feature in numerical_features:
            plt.figure(figsize=(10, 6))
            plot = sns.kdeplot(df[feature][(df['Churn'] == 0)], color="Red", shade=True)
            plot = sns.kdeplot(df[feature][(df['Churn'] == 1)], ax=plot, color="Blue", shade=True)
            plot.legend(["No Churn", "Churn"], loc='upper right')
            plot.set_ylabel('Density')
            plot.set_xlabel(feature)
            plot.set_title(f'{feature} by churn')
            st.pyplot(plt)
            st.write(df.groupby('Churn')[feature].describe())
            
    elif eda_option == "Correlation Matrix":
        plt.figure(figsize=(14, 10))
        correlation_matrix = df.drop(['customerID'], axis=1).corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Matrix')
        st.pyplot(plt)
        st.write(correlation_matrix)  # Summary statistics
    
    elif eda_option == "Gender Distribution":
        churn_by_gender = df.groupby(['gender', 'Churn']).size().unstack()
        churn_by_gender.plot(kind='bar', stacked=True, color=['lightblue', 'lightcoral'], figsize=(10, 6))
        plt.title('Churn Distribution by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Female', 'Male'], rotation=0)
        plt.legend(['No Churn', 'Churn'], loc='upper right')
        st.pyplot(plt)
        
    elif eda_option == "Payment Method Distribution":
    # Map numerical labels to payment method names
        payment_method_labels = {
        0: 'Electronic Check',
        1: 'Bank Transfer',
        2: 'Credit Card',
        3: 'Mailed Check'
    }

    # Count the churn distribution for each payment method
        churn_by_payment_method = df.groupby(['PaymentMethod', 'Churn']).size().unstack()

    # Rename the index to use payment method names
        churn_by_payment_method.index = churn_by_payment_method.index.map(payment_method_labels)

    # Plot the churn distribution for each payment method
        churn_by_payment_method.plot(kind='bar', stacked=True, figsize=(10, 6), color=['lightblue', 'lightcoral'])
        plt.title('Churn Distribution by Payment Method')
        plt.xlabel('Payment Method')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.legend(['No Churn', 'Churn'])
        st.pyplot(plt)



# Model Evaluation
elif option == "Model Evaluation":
    st.header("Model Evaluation")
    
    logreg_accuracy, logreg_precision, logreg_recall, logreg_f1, logreg_cm = evaluate_model(y_test, y_pred_logreg)
    rf_accuracy, rf_precision, rf_recall, rf_f1, rf_cm = evaluate_model(y_test, y_pred_rf)
    
    st.write("### Logistic Regression Confusion Matrix")
    st.write(logreg_cm)
    
    st.write("### Random Forest Confusion Matrix")
    st.write(rf_cm)
    
    data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Logistic Regression": [logreg_accuracy, logreg_precision, logreg_recall, logreg_f1],
        "Random Forest": [rf_accuracy, rf_precision, rf_recall, rf_f1]
    }
    
    st.table(pd.DataFrame(data))

# Check Churn
# Check Churn
# Check Churn
elif option == "Check Churn":
    st.header("Predict Customer Churn")
    
    st.write("Please enter the following details:")
    
    # User input fields
    gender = st.selectbox("Gender", ['Female', 'Male'])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ['No', 'Yes'])
    dependents = st.selectbox("Dependents", ['No', 'Yes'])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, step=1)
    phone_service = st.selectbox("Phone Service", ['No', 'Yes'])
    multiple_lines = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox("Online Backup", ['No', 'Yes', 'No internet service'])
    device_protection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox("Paperless Billing", ['No', 'Yes'])
    payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0)
    
    # Map user input
    user_data = {
        'gender': mappings['gender'][gender],
        'SeniorCitizen': senior_citizen,
        'Partner': mappings['Partner'][partner],
        'Dependents': mappings['Dependents'][dependents],
        'tenure': tenure,
        'PhoneService': mappings['PhoneService'][phone_service],
        'MultipleLines': mappings['MultipleLines'][multiple_lines],
        'InternetService': mappings['InternetService'][internet_service],
        'OnlineSecurity': mappings['OnlineSecurity'][online_security],
        'OnlineBackup': mappings['OnlineBackup'][online_backup],
        'DeviceProtection': mappings['DeviceProtection'][device_protection],
        'TechSupport': mappings['TechSupport'][tech_support],
        'StreamingTV': mappings['StreamingTV'][streaming_tv],
        'StreamingMovies': mappings['StreamingMovies'][streaming_movies],
        'Contract': mappings['Contract'][contract],
        'PaperlessBilling': mappings['PaperlessBilling'][paperless_billing],
        'PaymentMethod': mappings['PaymentMethod'][payment_method],
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'SeniorCitizen_Partner': senior_citizen * mappings['Partner'][partner]  # Include the derived feature
    }
    
    # Convert user input to dataframe
    user_df = pd.DataFrame(user_data, index=[0])
    
    # Standardize the numerical features
    user_df[numerical_columns] = scaler.transform(user_df[numerical_columns])
    
    # Predict churn
    if st.button("Check Churn"):
        churn_pred_logreg = logreg.predict(user_df)
        churn_pred_rf = rf.predict(user_df)
        
        # Use Random Forest prediction as final arbiter if models disagree
        if churn_pred_logreg[0] == churn_pred_rf[0]:
            final_churn = churn_pred_logreg[0]
        else:
            final_churn = churn_pred_rf[0]  # Trust Random Forest model
        
        st.write("### Prediction")
        st.write(f"Logistic Regression Prediction: {'Yes' if churn_pred_logreg[0] == 1 else 'No'}")
        st.write(f"Random Forest Prediction: {'Yes' if churn_pred_rf[0] == 1 else 'No'}")
        st.write(f"Final Churn Prediction: {'Yes' if final_churn == 1 else 'No'}")


