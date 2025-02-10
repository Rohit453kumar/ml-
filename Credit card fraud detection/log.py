import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score

def main():
    st.title("Credit Card Fraud Detection using Logistic Regression")
    
    # File Upload
    uploaded_file = st.file_uploader("creditcard", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.write(df.head())
        
        # Standardizing 'Time' and 'Amount'
        if 'Time' in df.columns and 'Amount' in df.columns:
            scaler = StandardScaler()
            df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
            st.write("### Standardized Data Preview")
            st.write(df.head())
        
        # Splitting data
        X = df.drop(columns=['Class']) if 'Class' in df.columns else df.drop(columns=[df.columns[-1]])
        y = df['Class'] if 'Class' in df.columns else df[df.columns[-1]]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Training Logistic Regression Model
        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Evaluation Metrics
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        st.write("### Model Performance")
        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write(f"**Recall:** {recall:.4f}")
        st.write(f"**F1 Score:** {f1:.4f}")
        
        # Classification Report
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
