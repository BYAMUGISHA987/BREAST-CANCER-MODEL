import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Define feature columns
feature_columns = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Header
st.title("Breast Cancer Tumor Prediction")
st.write("Predict whether a breast tumor is benign or malignant using a pre-trained Random Forest model (accuracy: 99.47%). Enter features manually or upload a CSV/Excel file.")

# Input method selection
input_method = st.radio("Choose input method:", ("Manual Input", "Upload File (CSV/Excel)"))

# Load model, scaler, and feature selector
@st.cache_resource
def load_model_and_scaler():
    try:
        model = load('wdbc_random_forest_model.joblib')
        scaler = load('wdbc_scaler.pkl')
        feature_indices = load('selected_features.joblib')  # Boolean mask or indices
        return model, scaler, feature_indices
    except FileNotFoundError:
        st.error("Model, scaler, or feature selection file not found. Ensure 'wdbc_random_forest_model.joblib', 'wdbc_scaler.pkl', and 'selected_features.joblib' are in the app directory.")
        st.stop()
model, scaler, feature_indices = load_model_and_scaler()

# Manual Input Section
if input_method == "Manual Input":
    st.header("Manual Feature Input")
    inputs = {}
    
    # Group features
    mean_features = [f for f in feature_columns if f.endswith('_mean')]
    se_features = [f for f in feature_columns if f.endswith('_se')]
    worst_features = [f for f in feature_columns if f.endswith('_worst')]
    
    # Mean Features
    with st.expander("Mean Features"):
        cols = st.columns(2)
        for i, feature in enumerate(mean_features):
            with cols[i % 2]:
                inputs[feature] = st.number_input(feature.replace('_', ' ').title(), min_value=0.0, value=0.0, step=0.01)
    
    # Standard Error Features
    with st.expander("Standard Error Features"):
        cols = st.columns(2)
        for i, feature in enumerate(se_features):
            with cols[i % 2]:
                inputs[feature] = st.number_input(feature.replace('_', ' ').title(), min_value=0.0, value=0.0, step=0.01)
    
    # Worst Features
    with st.expander("Worst Features"):
        cols = st.columns(2)
        for i, feature in enumerate(worst_features):
            with cols[i % 2]:
                inputs[feature] = st.number_input(feature.replace('_', ' ').title(), min_value=0.0, value=0.0, step=0.01)
    
    if st.button("Predict"):
        if all(v >= 0 for v in inputs.values()):
            # Prepare input
            X_input = np.array([inputs[f] for f in feature_columns]).reshape(1, -1)
            X_scaled = scaler.transform(X_input)
            X_selected = X_scaled[:, feature_indices]
            
            # Predict
            prediction = model.predict(X_selected)[0]
            probs = model.predict_proba(X_selected)[0]
            
            # Display results
            st.subheader("Prediction Result")
            st.write(f"Predicted: {'Malignant' if prediction == 1 else 'Benign'}")
            st.write(f"Probabilities: Benign={probs[0]:.4f}, Malignant={probs[1]:.4f}")
            
            # Bar chart
            fig = px.bar(
                x=['Benign', 'Malignant'],
                y=probs,
                labels={'x': 'Diagnosis', 'y': 'Probability'},
                color=['Benign', 'Malignant'],
                color_discrete_map={'Benign': '#1f77b4', 'Malignant': '#ff7f0e'}
            )
            st.plotly_chart(fig)
        else:
            st.error("All inputs must be non-negative numbers.")

# File Upload Section
else:
    st.header("Upload File")
    st.write("Upload a CSV or Excel file with the 30 features. Optional columns: 'id', 'diagnosis' (M/B or 1/0).")
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        # Read file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            # Validate columns
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.stop()
            
            # Prepare data
            X = df[feature_columns].values
            X_scaled = scaler.transform(X)
            X_selected = X_scaled[:, feature_indices]
            
            # Predict
            predictions = model.predict(X_selected)
            probs = model.predict_proba(X_selected)
            
            # Results DataFrame
            results = pd.DataFrame({
                'Prediction': ['Malignant' if p == 1 else 'Benign' for p in predictions],
                'Benign_Probability': probs[:, 0],
                'Malignant_Probability': probs[:, 1]
            })
            if 'id' in df.columns:
                results.insert(0, 'ID', df['id'])
            
            # Display results
            st.subheader("Prediction Results")
            st.write(f"Total samples: {len(results)}")
            st.write(f"Benign: {sum(results['Prediction'] == 'Benign')}, Malignant: {sum(results['Prediction'] == 'Malignant')}")
            st.dataframe(results)
            
            # Download results
            csv = results.to_csv(index=False)
            st.download_button("Download Results", csv, "predictions.csv", "text/csv")
            
            # Bar chart for aggregated probabilities
            avg_probs = [results['Benign_Probability'].mean(), results['Malignant_Probability'].mean()]
            fig = px.bar(
                x=['Benign', 'Malignant'],
                y=avg_probs,
                labels={'x': 'Diagnosis', 'y': 'Average Probability'},
                color=['Benign', 'Malignant'],
                color_discrete_map={'Benign': '#1f77b4', 'Malignant': '#ff7f0e'}
            )
            st.plotly_chart(fig)
            
            # Classification report if diagnosis is provided
            if 'diagnosis' in df.columns:
                y_true = df['diagnosis'].map({'M': 1, 'B': 0, 1: 1, 0: 0})
                from sklearn.metrics import classification_report
                report = classification_report(y_true, predictions, target_names=['Benign', 'Malignant'])
                st.subheader("Classification Report")
                st.text(report)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Model Info
st.sidebar.header("Model Information")
st.sidebar.write("Model: Random Forest")
st.sidebar.write("Accuracy: 99.47%")
st.sidebar.write("Features selected: 10 (using SelectKBest)")