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
# 禁用所有matplotlib字体相关警告
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
warnings.filterwarnings("ignore", message="findfont:*")

# --- Font Configuration ---
# Configure matplotlib to avoid font warnings
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
# 设置日志级别以减少字体警告
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# --- Page Configuration ---
st.set_page_config(
    page_title='ICU RRT Hypotension Risk Prediction',
    page_icon='💉',
    layout='wide'
)

# --- 简化的CSS样式 ---
st.markdown("""
<style>
    /* 主题色彩配置 */
    :root {
        --primary-color: #1E90FF;
        --success-color: #4CAF50;
        --warning-color: #FF9800;
        --error-color: #F44336;
        --card-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* 主容器样式 */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        background: #f8f9fa;
    }
    
    /* 标题样式 */
    .main-title {
        color: #1E90FF;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* 卡片样式 */
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: var(--card-shadow);
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
    }
    
    /* 指标卡片 */
    .metric-card {
        background: white;
        color: #333;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: var(--card-shadow);
        border: 2px solid #e0e0e0;
    }
    
    /* 风险等级颜色 */
    .risk-high {
        border-color: var(--error-color);
        background: #ffebee;
    }
    
    .risk-moderate {
        border-color: var(--warning-color);
        background: #fff8e1;
    }
    
    .risk-low {
        border-color: var(--success-color);
        background: #e8f5e8;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: var(--primary-color);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #1976D2;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
</style>
""", unsafe_allow_html=True)

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
    # Column mapping from CSV to model features
    column_mapping = {
        'Gender': 'gender',
        'Age': 'admission_age', 
        'Congestive_heart_failure': 'congestive_heart_failure',
        'Peripheral_vascular_disease': 'peripheral_vascular_disease',
        'Dementia': 'dementia',
        'Chronic_pulmonary_disease': 'chronic_pulmonary_disease',
        'Liver_disease': 'mild_liver_disease',
        'Diabetes': 'diabetes_without_cc',
        'Cancer': 'malignant_cancer',
        'Vasoactive_drugs': 'vasoactive_drugs',
        'PH': 'ph',
        'Lactate': 'lactate',
        'MAP': 'map',
        'SAP': 'sap',
        'ICU_to_RRT_initiation': 'icu_to_rrt_hours',
        'RRT_modality_IHD': 'rrt_type'
    }
    
    # Apply column mapping
    data = data.rename(columns=column_mapping)
    
    # Convert 'Yes'/'No' to binary
    binary_map = {'Yes': 1, 'No': 0}
    binary_cols = ['congestive_heart_failure', 'peripheral_vascular_disease', 'dementia', 
                   'chronic_pulmonary_disease', 'mild_liver_disease', 'diabetes_without_cc', 
                   'malignant_cancer', 'vasoactive_drugs']
    
    for col in binary_cols:
        if col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].map(binary_map)

    # Handle gender encoding
    if 'gender' in data.columns:
        # Handle both user input format (Male/Female) and CSV format (M/F)
        gender_map = {'M': 1, 'F': 0, 'Male': 1, 'Female': 0}
        data['gender'] = data['gender'].map(gender_map)

    # One-hot encode RRT type
    if 'rrt_type' in data.columns:
        # Handle both string format (CRRT/IHD) and numeric format (0/1)
        if data['rrt_type'].dtype == 'object':
            # Convert string to numeric: IHD=1, CRRT=0
            data['rrt_type'] = data['rrt_type'].map({'IHD': 1, 'CRRT': 0})
        data['rrt_type_IHD'] = (data['rrt_type'] == 1).astype(int)
        data = data.drop('rrt_type', axis=1)
    
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
    
    # Define input parameters for each feature (matching training features)
    input_params = [
        ('gender', 'Gender', 'selectbox', ('Male', 'Female'), None, None, None),
        ('admission_age', 'Age (years)', 'slider', 18, 100, 65, 1),
        ('congestive_heart_failure', 'Congestive Heart Failure', 'selectbox', ('Yes', 'No'), None, None, None),
        ('peripheral_vascular_disease', 'Peripheral Vascular Disease', 'selectbox', ('Yes', 'No'), None, None, None),
        ('dementia', 'Dementia', 'selectbox', ('Yes', 'No'), None, None, None),
        ('chronic_pulmonary_disease', 'Chronic Pulmonary Disease', 'selectbox', ('Yes', 'No'), None, None, None),
        ('mild_liver_disease', 'Mild Liver Disease', 'selectbox', ('Yes', 'No'), None, None, None),
        ('diabetes_without_cc', 'Diabetes without Complications', 'selectbox', ('Yes', 'No'), None, None, None),
        ('malignant_cancer', 'Malignant Cancer', 'selectbox', ('Yes', 'No'), None, None, None),
        ('vasoactive_drugs', 'Vasoactive Drugs', 'selectbox', ('Yes', 'No'), None, None, None),
        ('ph', 'Latest pH Value', 'slider', 7.00, 8.00, 7.40, 0.01),
        ('lactate', 'Latest Lactate Value (mmol/L)', 'slider', 0.0, 25.0, 2.0, 0.1),
        ('map', 'Mean Arterial Pressure (mmHg)', 'slider', 0, 250, 80, 1),
        ('sap', 'Systolic Arterial Pressure (mmHg)', 'slider', 0, 300, 120, 1),
        ('icu_to_rrt_hours', 'ICU to RRT Hours', 'slider', 0, 720, 24, 1),
        ('rrt_type', 'RRT Modality', 'selectbox', ('CRRT', 'IHD'), None, None, None),
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
        
        # Add SHAP value annotations
        for i, (idx, row) in enumerate(explanation_df.iterrows()):
            shap_text = f"{row['SHAP_Value']:.3f}"
            # Position SHAP value inside the bar
            x_pos = row['SHAP_Value'] / 2
            ax.text(x_pos, i, shap_text, 
                   va='center', ha='center',
                   fontsize=9, fontweight='bold', color='white')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info('''
        **SHAP Analysis Explanation:**
        - **Red bars (right side)**: Features increasing hypotension risk
        - **Green bars (left side)**: Features decreasing hypotension risk
        - Bar length and numbers show the magnitude of each feature's contribution to the final prediction
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
        
        # Add SHAP contribution annotations
        for i, (idx, row) in enumerate(feature_contributions.iterrows()):
            shap_text = f"{row['SHAP_Contribution']:.3f}"
            # Position SHAP value inside the bar
            x_pos = row['SHAP_Contribution'] / 2
            ax.text(x_pos, i, shap_text, 
                   va='center', ha='center',
                   fontsize=9, fontweight='bold', color='white')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown('''
        **Feature Contribution Analysis:**
        - Shows how each patient characteristic affects the hypotension risk prediction
        - **<font color='red'>Red bars (right side)</font>**: Features increasing hypotension risk
        - **<font color='green'>Green bars (left side)</font>**: Features decreasing hypotension risk
        - Numbers show the SHAP contribution values for each feature
        - Bar length represents the magnitude of impact on the final prediction
        ''', unsafe_allow_html=True)
        
    except Exception as e:
        st.info("Feature contribution analysis is temporarily unavailable. Please refer to the SHAP analysis above for detailed insights.")

# --- Main Program ---
def main():
    """Streamlit main function"""
    # 简洁的主标题
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="main-title">🏥 ICU RRT Hypotension Risk Prediction</h1>
        <p style="color: #666; font-size: 1.1rem; margin-bottom: 1rem;">
            AI-powered prediction system for hypotension risk assessment during renal replacement therapy
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    # 简化的模型信息部分
    st.markdown("""
    
    <div style="background: #fff8e1; padding: 1rem; border-radius: 8px; border-left: 4px solid #ff9800; margin-bottom: 1rem;">
        <p><strong>⚠️ Important Notes:</strong></p>
        <ul>
            <li>This model is designed for <strong>pre-RRT hypotension risk prediction only</strong></li>
            <li>Should <strong>not</strong> be used for selecting between IHD and CRRT modalities</li>
            <li>Results should be interpreted alongside clinical judgment and patient context</li>
        </ul>
    </div>
    
    <div style="background: #fce4ec; padding: 1rem; border-radius: 8px; border-left: 4px solid #e91e63;">
        <p><strong>⚖️ Medical Disclaimer:</strong> This AI prediction model is designed to <strong>assist, not replace</strong> clinical judgment. 
        It provides risk estimates based on historical data and identified risk factors, but does not guarantee 
        the actual occurrence or absence of hypotension. Always consult with qualified healthcare professionals 
        for medical decisions.</p>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar input
    with st.spinner("Loading input form..."):
        user_input_df = sidebar_input_features(feature_names)
    
    # 预测结果部分
    st.markdown("""
    <div class="prediction-card">
        <h2 style="color: #1E90FF; margin-bottom: 1rem;">🎯 Hypotension Risk Prediction</h2>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        prediction_proba = model.predict_proba(user_input_df)[0][1]
        
        # Unified risk level definition (≥73% is high risk)
        if prediction_proba >= 0.73:
            risk_level = "High Risk"
            risk_class = "risk-high"
            risk_icon = "🔴"
        elif prediction_proba >= 0.5:
            risk_level = "Moderate Risk"
            risk_class = "risk-moderate"
            risk_icon = "🟡"
        else:
            risk_level = "Low Risk"
            risk_class = "risk-low"
            risk_icon = "🟢"
        
        # 进度条显示
        # st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        st.markdown(f"**Hypotension Probability: {prediction_proba:.2%}**")
        st.progress(float(prediction_proba))
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 指标显示
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Probability</h3>
                <h2>{prediction_proba:.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card {risk_class}">
                <h3>{risk_icon} Risk Level</h3>
                <h2>{risk_level}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            confidence = "High" if abs(prediction_proba - 0.5) > 0.2 else "Medium"
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Confidence</h3>
                <h2>{confidence}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        # 风险解释
        st.markdown("<br>", unsafe_allow_html=True)
        if prediction_proba >= 0.73:
            st.error("🚨 **High Risk Alert**: This patient has a high probability of developing hypotension. Immediate preventive measures and close monitoring are strongly recommended.")
        elif prediction_proba >= 0.5:
            st.warning("⚠️ **Moderate Risk Alert**: This patient has some risk of hypotension. Enhanced monitoring and preparedness for intervention are recommended.")
        else:
            st.success("✅ **Low Risk**: This patient has a low risk of hypotension. Standard monitoring protocols are sufficient.")
            
    except Exception as e:
        st.error(f"Prediction error: {e}")
    
    # 特征重要性解释部分
    st.markdown("""
    <div class="prediction-card">
        <h2 style="color: #1E90FF; margin-bottom: 1rem;">📈 Feature Importance Analysis</h2>
        <p style="color: #666; margin-bottom: 1rem;">Understanding which factors contribute most to hypotension risk predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        display_global_explanations(model, X_train_processed, shap_image)
    
    # 个人预测解释部分
    st.markdown("""
    <div class="prediction-card">
        <h2 style="color: #1E90FF; margin-bottom: 1rem;">🔍 Individual Prediction Analysis</h2>
        <p style="color: #666; margin-bottom: 1rem;">How each patient characteristic influences the current prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        display_local_explanations(model, user_input_df, X_train_processed)

if __name__ == "__main__":
    main()