import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from PIL import Image
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
import pickle
from datetime import datetime
import streamlit.components.v1 as components
import io

# --- 中文字体配置 ---
# 解决Matplotlib中文显示问题
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except Exception:
    plt.rcParams['font.sans-serif'] = ['sans-serif'] # 回退到通用无衬线字体
    plt.rcParams['axes.unicode_minus'] = False
    st.warning("中文字体 'SimHei' 未找到，图表将使用默认英文字体显示。")

# --- 页面配置 ---
st.set_page_config(
    page_title='ICU肾脏替代治疗低血压风险预测',
    page_icon='💉',
    layout='wide'
)

# --- 资源加载 (使用缓存) ---
@st.cache_resource
def load_model(model_path):
    """加载XGBoost .json格式的模型文件"""
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

@st.cache_resource
def load_feature_names(feature_path):
    """加载模型特征列表"""
    with open(feature_path, 'rb') as f:
        features = pickle.load(f)
    return features

@st.cache_resource
def load_images(image_path):
    """加载图片"""
    return Image.open(image_path)

@st.cache_data
def load_training_data(data_path="train.csv"):
    """加载并缓存原始训练数据"""
    data = pd.read_csv(data_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    return data

def preprocess_data(data, feature_names):
    """对数据进行与训练时一致的预处理，用于解释器背景数据集"""
    # 转换 '是'/'否'
    binary_map = {'是': 1, '否': 0}
    for col in ['congestive_heart_failure', 'peripheral_vascular_disease', 'dementia', 
                'chronic_pulmonary_disease', 'mild_liver_disease', 'diabetes_without_cc', 
                'malignant_cancer', 'metastatic_solid_tumor', 'vasoactive_drugs']:
        if col in data.columns:
            # 训练数据已经是0/1，此步主要用于处理sidebar中来的原始数据
            if data[col].dtype == 'object':
                 data[col] = data[col].map(binary_map)

    # 对分类变量进行独热编码
    categorical_cols = ['gender', 'rrt_type']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # 使用 reindex 保证特征列完全对齐
    X = data.reindex(columns=feature_names, fill_value=0)
    return X

# --- 时间差计算函数 ---
def calculate_hours_diff(start_date, start_time, end_date, end_time):
    """计算两个日期时间之间的小时差"""
    start_dt = datetime.combine(start_date, start_time)
    end_dt = datetime.combine(end_date, end_time)
    diff = end_dt - start_dt
    return diff.total_seconds() / 3600

# --- UI 组件 ---
def sidebar_input_features(feature_names):
    """在侧边栏中创建用户输入组件"""
    st.sidebar.header('请在下方输入相关指标⬇️')
    
    # 初始化用户输入字典
    user_inputs = {}
    
    # 时间输入组件
    st.sidebar.subheader("时间计算")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.markdown("**入住ICU时间**")
        icu_date = st.date_input("日期", key="icu_date")
        icu_time = st.time_input("时间", key="icu_time")
    
    with col2:
        st.markdown("**肾脏替代治疗开始时间**")
        rrt_date = st.date_input("日期", key="rrt_date")
        rrt_time = st.time_input("时间", key="rrt_time")
    
    # 计算时间差
    icu_to_rrt_hours = calculate_hours_diff(icu_date, icu_time, rrt_date, rrt_time)
    st.sidebar.info(f"入住ICU到肾脏替代治疗开始时间差: **{icu_to_rrt_hours:.2f}小时**")
    
    # 特征输入组件
    st.sidebar.subheader("患者特征")
    
    # 定义每个特征的输入参数
    input_params = [
        ('gender', '性别', 'selectbox', ('男', '女'), None, None, None),
        ('admission_age', '年龄(岁)', 'slider', 18, 100, 60, 1),
        ('congestive_heart_failure', '合并充血性心力衰竭', 'selectbox', ('是', '否'), None, None, None),
        ('peripheral_vascular_disease', '合并外周血管疾病', 'selectbox', ('是', '否'), None, None, None),
        ('dementia', '合并痴呆', 'selectbox', ('是', '否'), None, None, None),
        ('chronic_pulmonary_disease', '合并慢性肺病', 'selectbox', ('是', '否'), None, None, None),
        ('mild_liver_disease', '合并轻度肝病', 'selectbox', ('是', '否'), None, None, None),
        ('diabetes_without_cc', '合并糖尿病', 'selectbox', ('是', '否'), None, None, None),
        ('malignant_cancer', '患有恶性肿瘤', 'selectbox', ('是', '否'), None, None, None),
        ('metastatic_solid_tumor', '转移性实体瘤', 'selectbox', ('是', '否'), None, None, None),
        ('vasoactive_drugs', '使用血管活性药物', 'selectbox', ('是', '否'), None, None, None),
        ('ph', '最近一次PH值', 'slider', 7.00, 8.00, 7.40, 0.01),
        ('lactate', '最近一次乳酸值(mmol/L)', 'slider', 0.0, 10.0, 2.0, 0.1),
        ('rrt_type', '肾脏替代治疗方式', 'selectbox', ('CRRT', 'IHD'), None, None, None),
        ('map', '治疗前平均动脉压(mmHg)', 'slider', 0, 250, 80, 1),
        ('sap', '治疗前收缩压(mmHg)', 'slider', 0, 250, 120, 1),
    ]
    
    # 创建输入组件
    for name, display, type, options, min_val, max_val, step in input_params:
        if type == 'slider':
            user_inputs[name] = st.sidebar.slider(display, min_value=min_val, max_value=max_val, value=options, step=step)
        elif type == 'selectbox':
            user_inputs[name] = st.sidebar.selectbox(display, options)
    
    # 添加计算的时间差
    user_inputs['icu_to_rrt_hours'] = icu_to_rrt_hours
    
    # 将用户输入构造成DataFrame
    input_df = pd.DataFrame([user_inputs])

    # 使用与训练数据相同的预处理流程
    output_df = preprocess_data(input_df, feature_names)

    return output_df

def display_global_explanations(model, X_train, shap_image):
    """显示全局模型解释（SHAP特征重要性图和依赖图）"""
    st.subheader("SHAP全局解释")

    # --- 计算SHAP值 ---
    with st.spinner("正在计算SHAP值，请稍候..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
    
    # 将SHAP值和原始数据转换为DataFrame
    shap_value_df = pd.DataFrame(shap_values, columns=X_train.columns)
    shap_data_df = X_train

    f1, f2 = st.columns(2)

    with f1:
        st.write('**SHAP特征重要性**')
        if shap_image:
            st.image(shap_image, use_container_width=True)
        else:
            st.warning("SHAP特征重要性图 ('shap.png') 未找到。请先运行 `generate_shap_image.py` 脚本。")
        st.info('SHAP特征重要性图显示了各个特征对模型输出的平均影响大小。它是通过计算数据集中每个特征的SHAP值的平均绝对值来排序的。条形越长，代表该特征对模型整体预测结果的影响越大。')

    with f2:
        st.write('**SHAP依赖图**')
        
        # 清理特征名以便显示
        feature_options = [name.replace('_', ' ') for name in shap_data_df.columns]
        feature_mapping = {clean: orig for clean, orig in zip(feature_options, shap_data_df.columns)}
        
        # 找出最重要的特征作为默认选项
        vals = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(feature_options, vals)), columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
        most_important_feature = feature_importance.iloc[0].col_name
        default_index = feature_options.index(most_important_feature) if most_important_feature in feature_options else 0
        
        selected_feature_cleaned = st.selectbox("选择变量", options=feature_options, index=default_index)
        
        # 将用户选择的清晰名称映射回原始列名
        selected_feature_orig = feature_mapping[selected_feature_cleaned]

        if selected_feature_orig in shap_value_df.columns:
            fig = px.scatter(
                x=shap_data_df[selected_feature_orig], 
                y=shap_value_df[selected_feature_orig], 
                color=shap_data_df[selected_feature_orig],
                color_continuous_scale=['blue','red'],
                labels={'x': f'{selected_feature_cleaned} 的原始值', 'y': 'SHAP值'}
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"特征 '{selected_feature_cleaned}' 的SHAP值不存在。")
        st.info('SHAP依赖图显示了单个变量对模型预测的影响。它说明了一个特征的每个值是如何影响预测结果的。')

def display_local_explanations(model, user_input_df, X_train):
    """显示局部模型解释（SHAP力图和LIME图）"""
    st.subheader("局部解释")
    
    # --- SHAP 力图 ---
    st.write('**SHAP力图**')
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(user_input_df)
        
        # 创建 SHAP 力图对象
        plot = shap.force_plot(
            explainer.expected_value,
            shap_values[0, :],
            user_input_df.iloc[0, :]
        )
        
        # 将图保存到内存中的HTML文件，确保JS被包含
        shap_html_path = io.StringIO()
        shap.save_html(shap_html_path, plot)
        
        # 从内存中读取HTML并显示
        components.html(shap_html_path.getvalue(), height=200)
        
        st.info('''
        **SHAP力图说明:**
        - 显示各特征如何将预测值从基础值推至最终预测值
        - 红色特征增加低血压风险
        - 蓝色特征降低低血压风险
        - 箭头长度表示影响大小
        ''')
    except Exception as e:
        st.error(f"生成SHAP力图时出错: {e}")

    # --- LIME 解释 ---
    st.write('**LIME解释**')
    try:
        # 确保 X_train 是 numpy array
        if isinstance(X_train, pd.DataFrame):
            X_train_values = X_train.values
        else:
            X_train_values = X_train

        # 将特征名中的下划线替换为空格，以提高可读性
        feature_names_cleaned = [name.replace('_', ' ') for name in X_train.columns]

        explainer = lime_tabular.LimeTabularExplainer(
            X_train_values, 
            feature_names=feature_names_cleaned, # 使用清理后的特征名
            class_names=['非低血压', '低血压'], 
            mode='classification',
            feature_selection='auto'
        )
        
        exp = explainer.explain_instance(
            user_input_df.values[0], 
            model.predict_proba, 
            num_features=10,
            labels=(1,) # 只解释"低血压"类别
        )
        
        # --- 手动生成LIME图以自定义颜色 ---
        # 红色代表增加风险，绿色代表降低风险
        exp_list = exp.as_list(label=1)
        
        # 检查是否获得了有效的解释
        if not exp_list:
            st.warning("LIME 未能为当前输入生成有效的解释。")
            return

        exp_list.reverse() # as_pyplot_figure会反转列表，在此模拟
        
        vals = [x[1] for x in exp_list]
        names = [x[0] for x in exp_list]
        
        # 交换颜色：正向影响（增加风险）为红色，负向为绿色
        colors = ['#d62728' if x > 0 else '#2ca02c' for x in vals]
        
        pos = np.arange(len(exp_list))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(pos, vals, align='center', color=colors)
        
        ax.set_yticks(pos)
        ax.set_yticklabels(names)
        ax.set_title('Local explanation for class 低血压')
        fig.tight_layout()
        
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown('''
        **LIME解释说明:**
        - 显示对当前预测影响最大的特征
        - **<font color='red'>红色条 (右侧)</font>**: 增加低血压风险的特征
        - **<font color='green'>绿色条 (左侧)</font>**: 降低低血压风险的特征
        - 数值表示对预测的具体影响
        ''', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"生成LIME图时出错: {e}")

# --- 主程序 ---
def main():
    """Streamlit主函数"""
    st.markdown("<h1 style='text-align: center; color: #1E90FF;'>ICU肾脏替代治疗低血压风险预测</h1>", unsafe_allow_html=True)
    
    # 加载资源
    shap_image = None
    try:
        model = load_model("hypotension_model.json")
        feature_names = load_feature_names("model_features.pkl")
        training_data = load_training_data("train.csv") # 确保train.csv在同级目录
        X_train_processed = preprocess_data(training_data.copy(), feature_names)
        try:
            shap_image = load_images("shap.png")
        except FileNotFoundError:
            st.warning("`shap.png` 文件未找到，SHAP特征重要性图将无法显示。请运行 `generate_shap_image.py` 脚本生成该文件。")

    except Exception as e:
        st.error(f"加载模型或数据时出错: {e}")
        st.error("请确保 `hypotension_model.json`, `model_features.pkl` 和 `train.csv` 文件位于应用根目录。")
        return
    
    # 显示模型信息
    st.info('''
    **关于模型:**
    - 预测目标: ICU患者进行肾脏替代治疗期间发生低血压的风险
    - 模型类型: XGBoost
    - 特征数量: 17个临床相关指标
    - 使用说明: 在左侧输入患者特征后，系统将实时计算低血压发生概率
    ''')
    
    # 侧边栏输入
    with st.spinner("加载输入表单..."):
        user_input_df = sidebar_input_features(feature_names)
    
    # 预测
    st.subheader('低血压风险预测')
    try:
        prediction_proba = model.predict_proba(user_input_df)[0][1]
        
        # 统一风险等级定义
        if prediction_proba > 0.7:
            risk_level = "高风险"
        elif prediction_proba > 0.5:
            risk_level = "中风险"
        else:
            risk_level = "低风险"
        
        # 创建进度条 (将numpy.float32转换为float)
        st.progress(float(prediction_proba))
        
        # 显示预测结果
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="低血压发生概率", value=f"{prediction_proba:.2%}")
        with col2:
            st.metric(label="风险等级", value=risk_level)
            
        # 风险解释
        if prediction_proba > 0.7:
            st.warning("⚠️ 高风险预警: 该患者发生低血压的可能性很高，建议采取预防措施")
        elif prediction_proba > 0.5:
            st.warning("⚠️ 中风险预警: 该患者有一定低血压风险，建议密切监测")
        else:
            st.success("✅ 低风险: 该患者低血压风险较低")
            
    except Exception as e:
        st.error(f"预测出错: {e}")
    
    # 特征重要性解释
    st.subheader("特征重要性解释")
    display_global_explanations(model, X_train_processed, shap_image)
    
    # 局部解释
    st.subheader("当前预测解释")
    display_local_explanations(model, user_input_df, X_train_processed)

if __name__ == "__main__":
    main()