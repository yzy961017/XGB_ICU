import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import pickle

def preprocess_data(data: pd.DataFrame):
    """对数据进行预处理，与 train.py 中的逻辑保持完全一致"""
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)

    # --- 列名映射：将实际CSV列名映射为模型期望的列名 ---
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
    
    # 应用列名映射
    data = data.rename(columns=column_mapping)

    # 处理vasoactive.drugs列名中的点号
    if 'vasoactive.drugs' in data.columns:
        data = data.rename(columns={'vasoactive.drugs': 'vasoactive_drugs'})

    # 将性别转换为数值 (M=1, F=0)
    if 'gender' in data.columns:
        data['gender'] = data['gender'].map({'M': 1, 'F': 0})
    
    # 对RRT类型进行独热编码
    if 'rrt_type' in data.columns:
        data = pd.get_dummies(data, columns=['rrt_type'], prefix='rrt_type', drop_first=True, dtype=int)
    
    return data

def generate_shap_summary_plot(model_path, data_path, feature_list_path, output_path):
    """
    加载GBM模型和测试数据，生成并保存SHAP摘要图（蜂窝散点图）。
    """
    try:
        # 设置字体以避免警告
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        # 禁用字体警告
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    except Exception:
        print("警告: 字体配置失败，使用默认设置。")

    print("加载GBM模型...")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"模型加载成功: {type(model)}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    print("加载特征列表...")
    try:
        feature_names = joblib.load(feature_list_path)
        print(f"特征列表加载成功，共 {len(feature_names)} 个特征: {feature_names}")
    except Exception as e:
        print(f"特征列表加载失败: {e}")
        return

    print(f"加载并预处理测试数据 (来自 {data_path})...")
    try:
        data = pd.read_csv(data_path)
        print(f"原始数据形状: {data.shape}")
        processed_data = preprocess_data(data)
        print(f"预处理后数据形状: {processed_data.shape}")
        print(f"预处理后列名: {list(processed_data.columns)}")
        
        # 使用 reindex 保证测试数据的列与训练时完全一致
        X_test = processed_data.reindex(columns=feature_names, fill_value=0)
        print(f"最终特征矩阵形状: {X_test.shape}")
        
        # 检查是否有缺失值
        if X_test.isnull().any().any():
            print("警告: 特征矩阵中存在缺失值")
            X_test = X_test.fillna(0)
            
    except Exception as e:
        print(f"数据加载和预处理失败: {e}")
        return

    print("计算SHAP值...")
    try:
        # 对于GBM模型，使用TreeExplainer
        explainer = shap.TreeExplainer(model)
        # 为了计算速度，只使用前100个样本
        sample_size = min(100, len(X_test))
        X_sample = X_test.iloc[:sample_size]
        print(f"使用 {sample_size} 个样本计算SHAP值")
        
        shap_values = explainer.shap_values(X_sample)
        print(f"SHAP值计算完成，形状: {shap_values.shape}")
        
    except Exception as e:
        print(f"SHAP值计算失败: {e}")
        return

    print("生成SHAP摘要图（蜂窝图）...")
    try:
        # 为了在图上显示更清晰的标签，创建显示用的列名
        display_names = [col.replace('_', ' ').title() for col in feature_names]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X_sample, 
            feature_names=display_names,
            plot_type="dot", 
            show=False, 
            max_display=len(feature_names)
        )
        
        fig = plt.gcf()
        fig.tight_layout()
        
        print(f"保存图片到 {output_path}...")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("SHAP图像生成完成！")
        
    except Exception as e:
        print(f"SHAP图像生成失败: {e}")
        return

if __name__ == "__main__":
    MODEL_FILE = "hypotension_model.pkl"  # 更新为pkl格式
    DATA_FILE = "test.csv"  # 使用测试集来生成SHAP图
    FEATURES_FILE = "model_features.pkl"
    OUTPUT_IMAGE_FILE = "shap.png"
    
    generate_shap_summary_plot(MODEL_FILE, DATA_FILE, FEATURES_FILE, OUTPUT_IMAGE_FILE)