import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.title("🩺 Breast Cancer Prediction App")

# Sidebar for navigation
section = st.sidebar.radio("Navigate to", ["Dataset", "Manual Input", "Visualizations", "Q&A"])

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("breast_cancer_data.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset file 'breast_cancer_data.csv' not found. Please place it in the app folder.")
    st.stop()

if 'target' not in df.columns:
    st.error("Dataset must have a 'target' column (0=malignant, 1=benign).")
    st.stop()

features = [c for c in df.columns if c != "target"]

# Preprocess & train model once
X = df[features]
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# Section: Dataset
if section == "Dataset":
    st.header("Dataset Preview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df)
    st.markdown(f"Model Accuracy on Test Set: **{accuracy*100:.2f}%**")

# Section: Manual Input
elif section == "Manual Input":
    st.header("Enter Patient Data")

    user_input = {}
    for feature in features:
        val = st.number_input(f"{feature}", value=float(df[feature].mean()))
        user_input[feature] = val

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        label = "Benign" if prediction == 1 else "Malignant"
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence (Benign): {proba[1]*100:.2f}%")

# Section: Visualizations
elif section == "Visualizations":
    st.header("Data Visualizations")

    st.subheader("Target Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="target", data=df, palette="Set2", ax=ax)
    ax.set_xticklabels(["Malignant", "Benign"])
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Section: Q&A
elif section == "Q&A":
    st.header("Breast Cancer Awareness & Recommendations")

    st.subheader("What is breast cancer?")
    st.write("Breast cancer is uncontrolled growth of abnormal breast cells.")

    st.subheader("Signs & Symptoms")
    st.markdown("""
    - Lump in breast or underarm  
    - Change in breast size or shape  
    - Unusual discharge  
    - Skin dimpling or redness  
    """)

    st.subheader("Recommendations")
    st.markdown("""
    - Regular self-exams  
    - Annual check-ups  
    - Healthy lifestyle  
    - Follow doctor advice  
    """)

st.markdown("---")
st.markdown("Made with ❤️ by oct")
