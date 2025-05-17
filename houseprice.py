import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Try importing xgboost, show installation instructions if not found
try:
    import xgboost
except ImportError:
    st.error("XGBoost is not installed. Please install it using: pip install xgboost")
    st.markdown("""
    ```
    pip install xgboost
    ```
    Then restart this application.
    """)
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("House Price Prediction App")
st.markdown("""
This app predicts house prices based on various features using an XGBoost model.
Enter the values for each feature to get a prediction.
""")

@st.cache_resource
def load_model():
    """Load the trained model from pickle file"""
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the saved model
try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.warning("Please make sure the 'model.pkl' file is in the same directory as this app.")
    st.stop()

# Create feature input sections
st.header("Input Features")

# Use columns to organize the layout
col1, col2, col3 = st.columns(3)

with col1:
    crim = st.number_input("CRIM (Per capita crime rate)", min_value=0.0, format="%.6f", help="Per capita crime rate by town")
    zn = st.number_input("ZN (Residential land zoned)", min_value=0.0, format="%.1f", help="Proportion of residential land zoned for lots over 25,000 sq.ft.")
    indus = st.number_input("INDUS (Non-retail business acres)", min_value=0.0, format="%.2f", help="Proportion of non-retail business acres per town")
    chas = st.selectbox("CHAS (Charles River dummy variable)", options=[0, 1], help="Charles River dummy variable (1 if tract bounds river; 0 otherwise)")
    nox = st.number_input("NOX (Nitric oxides concentration)", min_value=0.0, max_value=1.0, format="%.3f", help="Nitric oxides concentration (parts per 10 million)")

with col2:
    rm = st.number_input("RM (Average number of rooms)", min_value=0.0, format="%.2f", help="Average number of rooms per dwelling")
    age = st.number_input("AGE (Proportion of old units)", min_value=0.0, max_value=100.0, format="%.1f", help="Proportion of owner-occupied units built prior to 1940")
    dis = st.number_input("DIS (Distances to employment centers)", min_value=0.0, format="%.2f", help="Weighted distances to five Boston employment centers")
    rad = st.number_input("RAD (Accessibility to highways)", min_value=0, format="%d", help="Index of accessibility to radial highways")
    tax = st.number_input("TAX (Property tax rate)", min_value=0, format="%d", help="Full-value property tax rate per $10,000")

with col3:
    ptratio = st.number_input("PTRATIO (Pupil-teacher ratio)", min_value=0.0, format="%.1f", help="Pupil-teacher ratio by town")
    b = st.number_input("B (Black population ratio)", min_value=0.0, format="%.2f", help="1000(Bk - 0.63)^2 where Bk is the proportion of Black people by town")
    lstat = st.number_input("LSTAT (Lower status population)", min_value=0.0, format="%.2f", help="% lower status of the population")

# Create a feature dictionary
features = {
    'CRIM': crim,
    'ZN': zn,
    'INDUS': indus,
    'CHAS': chas,
    'NOX': nox,
    'RM': rm,
    'AGE': age,
    'DIS': dis,
    'RAD': rad,
    'TAX': tax,
    'PTRATIO': ptratio,
    'B': b,
    'LSTAT': lstat
}

# Create a DataFrame with the user inputs
input_df = pd.DataFrame([features])

# Prediction button
predict_button = st.button("Predict House Price")

if predict_button:
    try:
        # Make prediction
        prediction = model.predict(input_df)
        
        # Display prediction
        st.header("Prediction")
        st.subheader(f"The predicted house price (MEDV) is: ${prediction[0] * 1000:.2f}")
        
        # Add confidence note
        st.info("Note: This prediction is based on the Boston Housing dataset. The target variable MEDV is in $1000's, so the displayed result has been converted to USD.")
        
        # Show input summary
        st.subheader("Input Summary")
        st.dataframe(input_df)
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.warning("Please check your input values and try again.")

# Add sidebar with additional information
st.sidebar.header("About")
st.sidebar.markdown("""
## House Price Prediction Model
This application uses an XGBoost Regressor model trained on the Boston Housing dataset.

### Features:
- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxides concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centers
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of Black people by town
- **LSTAT**: % lower status of the population
- **MEDV**: Median value of owner-occupied homes in $1000's (Target variable)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ using Streamlit and XGBoost")
