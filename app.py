import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Hide Streamlit style elements: menu, footer, header
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Load dataset and keep only first 5 features
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names).iloc[:, :5]
y = data.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Logistic Regression model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit app layout
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")
st.title("🩺 Breast Cancer Prediction App")
st.write("Upload patient data or enter values manually to predict if the tumor is **malignant (cancerous)** or **benign (non-cancerous)**.")

# Image upload section
st.subheader("📷 Optional: Upload Scan Image")
uploaded_image = st.file_uploader("Upload Image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

# Data upload section (CSV or Excel)
st.subheader("📄 Optional: Upload Data File (CSV or Excel)")
uploaded_file = st.file_uploader("Upload a CSV or Excel file with required features", type=["csv", "xls", "xlsx"])

input_df = pd.DataFrame()

if uploaded_file:
    try:
        if uploaded_file.name.endswith(('xls', 'xlsx')):
            input_df = pd.read_excel(uploaded_file)
        else:
            input_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Failed to read file: {e}")

# Manual input fields if no file uploaded or file is empty
if input_df.empty:
    st.subheader("📝 Manual Data Entry")

    def user_input_features():
        mean_radius = st.number_input('Mean Radius', float(X['mean radius'].min()), float(X['mean radius'].max()))
        mean_texture = st.number_input('Mean Texture', float(X['mean texture'].min()), float(X['mean texture'].max()))
        mean_perimeter = st.number_input('Mean Perimeter', float(X['mean perimeter'].min()), float(X['mean perimeter'].max()))
        mean_area = st.number_input('Mean Area', float(X['mean area'].min()), float(X['mean area'].max()))
        mean_smoothness = st.number_input('Mean Smoothness', float(X['mean smoothness'].min()), float(X['mean smoothness'].max()))

        data_input = {
            'mean radius': mean_radius,
            'mean texture': mean_texture,
            'mean perimeter': mean_perimeter,
            'mean area': mean_area,
            'mean smoothness': mean_smoothness
        }
        return pd.DataFrame(data_input, index=[0])

    input_df = user_input_features()

# Prediction
if not input_df.empty:
    # Make sure input columns match training columns (5 features)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader('Prediction')
    st.write('Malignant (Cancer)' if prediction[0] == 0 else 'Benign (No Cancer)')

    st.subheader('Prediction Probability')
    st.write(f"Benign: {prediction_proba[0][1]*100:.2f}%")
    st.write(f"Malignant: {prediction_proba[0][0]*100:.2f}%")
else:
    st.warning("Please upload a data file or enter data manually to get predictions.")
