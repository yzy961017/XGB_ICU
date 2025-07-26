import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from PIL import Image
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
import joblib
from datetime import datetime
import streamlit.components.v1 as components
import io
import pickle
import warnings
warnings.filterwarnings("ignore", message="findfont: Generic family 'sans-serif' not found*")
# --- Font Configuration ---
# Configure matplotlib for English display
plt.rcParams['font.sans-serif'] = ['sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- Page Configuration ---
st.set_page_config(
    page_title='ICU RRT Hypotension Risk Prediction',
    page_icon='💉',
    layout='wide'
)

# --- 资源加载 (使用缓存) ---
@st.cache_resource
def load_model(model_path):
    """Load GBM model from pickle format"""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_feature_names(feature_path):
    """Load model feature list"""
    features = joblib.load(feature_path)
    return features

@st.cache_resource
def load_images(image_path):
    """Load images"""
    return Image.open(image_path)

@st.cache_data
def load_training_data(data_path="train.csv"):
    """Load and cache original training data"""
    data = pd.read_csv(data_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    return data

def preprocess_data(data, feature_names):
    """Preprocess data consistent with training for explainer background dataset"""
    # Convert 'Yes'/'No' to binary
    binary_map = {'Yes': 1, 'No': 0}
    binary_cols = ['Congestive_heart_failure', 'Peripheral_vascular_disease', 'Dementia', 
                   'Chronic_pulmonary_disease', 'Liver_disease', 'Diabetes', 
                   'Cancer', 'vasoactive_drugs']
    
    for col in binary_cols:
        if col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].map(binary_map)

    # Handle gender encoding
    if 'gender' in data.columns:
        data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

    # One-hot encode RRT type
    if 'RRT_modality_IHD' in data.columns:
        data['RRT_modality_IHD'] = (data['RRT_modality_IHD'] == 'IHD').astype(int)
        data = data.drop('RRT_modality_IHD', axis=1)
    
    # Ensure feature alignment
    X = data.reindex(columns=feature_names, fill_value=0)
    return X

# --- Time Difference Calculation Function ---
def calculate_hours_diff(start_date, start_time, end_date, end_time):
    """Calculate hour difference between two datetime points"""
    start_dt = datetime.combine(start_date, start_time)
    end_dt = datetime.combine(end_date, end_time)
    diff = end_dt - start_dt
    return diff.total_seconds() / 3600

# --- UI Components ---
def sidebar_input_features(feature_names):
    """Create user input components in sidebar"""
    st.sidebar.header('Please enter patient characteristics below ⬇️')
    
    # Initialize user input dictionary
    user_inputs = {}
    
    # Time input components
    st.sidebar.subheader("Time Calculation")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.markdown("**ICU Admission Time**")
        icu_date = st.date_input("Date", key="icu_date")
        icu_time = st.time_input("Time", key="icu_time")
    
    with col2:
        st.markdown("**RRT Start Time**")
        rrt_date = st.date_input("Date", key="rrt_date")
        rrt_time = st.time_input("Time", key="rrt_time")
    
    # Calculate time difference
    icu_to_rrt_hours = calculate_hours_diff(icu_date, icu_time, rrt_date, rrt_time)
    st.sidebar.info(f"ICU admission to RRT start time difference: **{icu_to_rrt_hours:.2f} hours**")
    
    # Feature input components
    st.sidebar.subheader("Patient Characteristics")
    
    # Define input parameters for each feature
    input_params = [
        ('Gender', 'Gender', 'selectbox', ('Male', 'Female'), None, None, None),
        ('Age', 'Age (years)', 'slider', 18, 100, 65, 1),
        ('Congestive_heart_failure', 'Congestive Heart Failure', 'selectbox', ('Yes', 'No'), None, None, None),
        ('Peripheral_vascular_disease', 'Peripheral Vascular Disease', 'selectbox', ('Yes', 'No'), None, None, None),
        ('Dementia', 'Dementia', 'selectbox', ('Yes', 'No'), None, None, None),
        ('Chronic_pulmonary_disease', 'Chronic Pulmonary Disease', 'selectbox', ('Yes', 'No'), None, None, None),
        ('Liver_disease', 'Mild Liver Disease', 'selectbox', ('Yes', 'No'), None, None, None),
        ('Diabetes', 'Diabetes', 'selectbox', ('Yes', 'No'), None, None, None),
        ('Cancer', 'Cancer', 'selectbox', ('Yes', 'No'), None, None, None),
        ('vasoactive_drugs', 'Vasoactive Drugs', 'selectbox', ('Yes', 'No'), None, None, None),
        ('PH', 'Latest pH Value', 'slider', 7.00, 8.00, 7.40, 0.01),
        ('Lactate', 'Latest Lactate Value (mmol/L)', 'slider', 0.0, 25.0, 2.0, 0.1),
        ('RRT_modality_IHD', 'RRT Modality', 'selectbox', ('CRRT', 'IHD'), None, None, None),
        ('SAP', 'SAP (mmHg)', 'slider', 0, 250, 80, 1),
        ('MAP', 'MAP (mmHg)', 'slider', 0, 250, 80, 1),
    ]
    
    # Create input components
    for name, display, type, p1, p2, p3, p4 in input_params:
        if type == 'slider':
            # p1: min_val, p2: max_val, p3: default_val, p4: step
            user_inputs[name] = st.sidebar.slider(display, min_value=p1, max_value=p2, value=p3, step=p4)
        elif type == 'selectbox':
            # p1: options
            user_inputs[name] = st.sidebar.selectbox(display, p1)
    
    # Add calculated time difference
    user_inputs['icu_to_rrt_hours'] = icu_to_rrt_hours
    
    # Convert user inputs to DataFrame
    input_df = pd.DataFrame([user_inputs])

    # Use same preprocessing pipeline as training data
    output_df = preprocess_data(input_df, feature_names)

    return output_df

def display_global_explanations(model, X_train, shap_image):
    """Display global model explanations (SHAP feature importance and dependence plots)"""
    st.subheader("SHAP Global Explanations")

    # --- Calculate SHAP values ---
    with st.spinner("Calculating SHAP values, please wait..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
    
    # Convert SHAP values and original data to DataFrame
    shap_value_df = pd.DataFrame(shap_values, columns=X_train.columns)
    shap_data_df = X_train

    f1, f2 = st.columns(2)

    with f1:
        st.write('**SHAP Feature Importance**')
        if shap_image:
            st.image(shap_image, use_container_width=True)
        else:
            st.warning("SHAP feature importance plot ('shap.png') not found. Please run `generate_shap_image.py` script first.")
        st.info('The SHAP feature importance plot shows the average impact of each feature on model output. It is ranked by calculating the mean absolute SHAP values for each feature in the dataset. Longer bars indicate greater influence on overall model predictions.')

    with f2:
        st.write('**SHAP Dependence Plot**')
        
        # Clean feature names for display
        feature_options = [name.replace('_', ' ').title() for name in shap_data_df.columns]
        feature_mapping = {clean: orig for clean, orig in zip(feature_options, shap_data_df.columns)}
        
        # Find most important feature as default option
        vals = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(feature_options, vals)), columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
        most_important_feature = feature_importance.iloc[0].col_name
        default_index = feature_options.index(most_important_feature) if most_important_feature in feature_options else 0
        
        selected_feature_cleaned = st.selectbox("Select Variable", options=feature_options, index=default_index)
        
        # Map user-selected clean name back to original column name
        selected_feature_orig = feature_mapping[selected_feature_cleaned]

        if selected_feature_orig in shap_value_df.columns:
            fig = px.scatter(
                x=shap_data_df[selected_feature_orig], 
                y=shap_value_df[selected_feature_orig], 
                color=shap_data_df[selected_feature_orig],
                color_continuous_scale=['blue','red'],
                labels={'x': f'{selected_feature_cleaned} Original Values', 'y': 'SHAP Values'}
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"SHAP values for feature '{selected_feature_cleaned}' do not exist.")
        st.info('The SHAP dependence plot shows how a single variable affects model predictions. It illustrates how each value of a feature influences the prediction outcome.')

def display_local_explanations(model, user_input_df, X_train):
    """Display local model explanations (SHAP force plot and LIME plot)"""
    st.subheader("Local Explanations")
    
    # --- SHAP Force Plot ---
    st.write('**SHAP Force Plot**')
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(user_input_df)
        
        # Handle different SHAP output formats for GBM
        if isinstance(shap_values, list):
            # For binary classification, use class 1 (positive class)
            shap_values_to_plot = shap_values[1][0, :]
            expected_value = explainer.expected_value[1]
        else:
            # For single output
            shap_values_to_plot = shap_values[0, :]
            expected_value = explainer.expected_value
        
        # Display precise probability values
        prediction_proba = model.predict_proba(user_input_df)[0][1]
        # Convert numpy types to Python float to avoid format string errors
        prediction_proba_float = float(prediction_proba)
        expected_value_float = float(expected_value)
        
        st.write(f"**Current input prediction probability:** `{prediction_proba_float:.4f}`")
        st.write(f"**Model baseline probability (expected value):** `{expected_value_float:.4f}`")
        
        # Create a simple waterfall-style explanation instead of force plot
        feature_names = user_input_df.columns.tolist()
        feature_values = user_input_df.iloc[0].values
        
        # Create explanation dataframe
        explanation_df = pd.DataFrame({
            'Feature': [name.replace('_', ' ').title() for name in feature_names],
            'Value': feature_values,
            'SHAP_Value': shap_values_to_plot
        })
        
        # Sort by absolute SHAP value
        explanation_df['Abs_SHAP'] = np.abs(explanation_df['SHAP_Value'])
        explanation_df = explanation_df.sort_values('Abs_SHAP', ascending=False).head(10)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#d62728' if x > 0 else '#2ca02c' for x in explanation_df['SHAP_Value']]
        
        bars = ax.barh(range(len(explanation_df)), explanation_df['SHAP_Value'], color=colors)
        ax.set_yticks(range(len(explanation_df)))
        ax.set_yticklabels(explanation_df['Feature'])
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title('SHAP Feature Impact Analysis')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value annotations for SHAP values
        for i, (idx, row) in enumerate(explanation_df.iterrows()):
            value_text = f"Value: {row['Value']:.2f}"
            ax.text(0.02 if row['SHAP_Value'] > 0 else -0.02, i, value_text, 
                   va='center', ha='left' if row['SHAP_Value'] > 0 else 'right',
                   fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info('''
        **SHAP Analysis Explanation:**
        - **Red bars (right side)**: Features increasing hypotension risk
        - **Green bars (left side)**: Features decreasing hypotension risk
        - Bar length indicates the magnitude of feature influence
        - Values show how much each feature contributes to the final prediction
        ''')
    except Exception as e:
        st.error(f"Error generating SHAP analysis: {e}")

    # --- Alternative Local Explanation ---
    st.write('**Feature Contribution Analysis**')
    try:
        # Create a simplified feature contribution analysis
        # This provides similar insights to LIME but with better numerical stability
        
        # Get feature names and values
        feature_names = user_input_df.columns.tolist()
        feature_values = user_input_df.iloc[0].values
        
        # Calculate feature contributions using SHAP values we already computed
        feature_contributions = pd.DataFrame({
            'Feature': [name.replace('_', ' ').title() for name in feature_names],
            'Value': feature_values,
            'SHAP_Contribution': shap_values_to_plot
        })
        
        # Sort by absolute contribution
        feature_contributions['Abs_Contribution'] = np.abs(feature_contributions['SHAP_Contribution'])
        feature_contributions = feature_contributions.sort_values('Abs_Contribution', ascending=False).head(8)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#d62728' if x > 0 else '#2ca02c' for x in feature_contributions['SHAP_Contribution']]
        
        bars = ax.barh(range(len(feature_contributions)), feature_contributions['SHAP_Contribution'], color=colors)
        ax.set_yticks(range(len(feature_contributions)))
        ax.set_yticklabels(feature_contributions['Feature'])
        ax.set_xlabel('Feature Contribution to Prediction')
        ax.set_title('Individual Feature Impact on Current Prediction')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value annotations
        for i, (idx, row) in enumerate(feature_contributions.iterrows()):
            value_text = f"Value: {row['Value']:.2f}"
            ax.text(0.02 if row['SHAP_Contribution'] > 0 else -0.02, i, value_text, 
                   va='center', ha='left' if row['SHAP_Contribution'] > 0 else 'right',
                   fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown('''
        **Feature Contribution Analysis:**
        - Shows how each patient characteristic affects the hypotension risk prediction
        - **<font color='red'>Red bars (right side)</font>**: Features increasing hypotension risk
        - **<font color='green'>Green bars (left side)</font>**: Features decreasing hypotension risk
        - "Value" shows the actual patient measurement for each feature
        - Bar length represents the magnitude of impact on the final prediction
        ''', unsafe_allow_html=True)
        
    except Exception as e:
        st.info("Feature contribution analysis is temporarily unavailable. Please refer to the SHAP analysis above for detailed insights.")

# --- Main Program ---
def main():
    """Streamlit main function"""
    st.markdown("<h1 style='text-align: center; color: #1E90FF;'>ICU RRT Hypotension Risk Prediction</h1>", unsafe_allow_html=True)
    
    # Load resources
    shap_image = None
    try:
        model = load_model("hypotension_model.pkl")
        feature_names = load_feature_names("model_features.pkl")
        training_data = load_training_data("train.csv") # Ensure train.csv is in same directory
        X_train_processed = preprocess_data(training_data.copy(), feature_names)
        try:
            shap_image = load_images("shap.png")
        except FileNotFoundError:
            st.warning("`shap.png` file not found, SHAP feature importance plot will not be displayed. Please run `generate_shap_image.py` script to generate this file.")

    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        st.error("Please ensure `hypotension_model.pkl`, `model_features.pkl` and `train.csv` files are in the application root directory.")
        return
    
    # Display model information
    st.info('''
    **About the Model:**
    • **Prediction Target**: Risk of hypotension during renal replacement therapy (RRT) in ICU patients
    • **Model Type**: Gradient Boosting Machine (GBM)
    • **Instructions for Use**: After entering patient characteristics on the left panel, the system will calculate the real-time probability of hypotension.
    
    ⚠️ **Note**: This model is intended solely for pre-RRT prediction of hypotension risk. It should not be used as a basis for selecting between IHD and CRRT modalities.
    
    ⚠️ **Disclaimer**: This prediction model is designed to assist, not replace, clinical judgment. It estimates the risk of hypotension based on historical data and identified risk factors, but it does not guarantee the actual occurrence or absence of hypotension.
    ''')
    
    # Sidebar input
    with st.spinner("Loading input form..."):
        user_input_df = sidebar_input_features(feature_names)
    
    # Prediction
    st.subheader('Hypotension Risk Prediction')
    try:
        prediction_proba = model.predict_proba(user_input_df)[0][1]
        
        # Unified risk level definition (≥73% is high risk)
        if prediction_proba >= 0.73:
            risk_level = "High Risk"
        elif prediction_proba >= 0.5:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"
        
        # Create progress bar (convert numpy.float32 to float)
        st.progress(float(prediction_proba))
        
        # Display prediction results
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Hypotension Probability", value=f"{prediction_proba:.2%}")
        with col2:
            st.metric(label="Risk Level", value=risk_level)
            
        # Risk interpretation
        if prediction_proba >= 0.73:
            st.warning("⚠️ High Risk Alert: This patient has a high probability of developing hypotension. Preventive measures are recommended.")
        elif prediction_proba >= 0.5:
            st.warning("⚠️ Moderate Risk Alert: This patient has some risk of hypotension. Close monitoring is recommended.")
        else:
            st.success("✅ Low Risk: This patient has a low risk of hypotension.")
            
    except Exception as e:
        st.error(f"Prediction error: {e}")
    
    # Feature importance explanation
    st.subheader("Feature Importance Explanation")
    display_global_explanations(model, X_train_processed, shap_image)
    
    # Local explanation
    st.subheader("Current Prediction Explanation")
    display_local_explanations(model, user_input_df, X_train_processed)

if __name__ == "__main__":
    main()
