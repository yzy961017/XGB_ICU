import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import pickle

def preprocess_data(data: pd.DataFrame):
    """对数据进行预处理，与 train.py 中的逻辑保持完全一致"""
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)

    if 'vasoactive.drugs' in data.columns:
        data = data.rename(columns={'vasoactive.drugs': 'vasoactive_drugs'})

    binary_map = {'是': 1, '否': 0}
    binary_cols = [
        'congestive_heart_failure', 'peripheral_vascular_disease', 'dementia', 
        'chronic_pulmonary_disease', 'mild_liver_disease', 'diabetes_without_cc', 
        'malignant_cancer', 'metastatic_solid_tumor', 'vasoactive_drugs'
    ]
    for col in binary_cols:
        if col in data.columns and data[col].dtype == 'object':
            data[col] = data[col].map(binary_map)

    categorical_cols = ['gender', 'rrt_type']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True, dtype=int)
    
    return data

def generate_shap_summary_plot(model_path, data_path, feature_list_path, output_path):
    """
    加载模型和测试数据，生成并保存SHAP摘要图（蜂窝散点图）。
    """
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        print("警告: 未找到中文字体'SimHei'，图表可能无法正确显示中文。")

    print("加载模型...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    print("加载特征列表...")
    with open(feature_list_path, 'rb') as f:
        feature_names = pickle.load(f)

    print(f"加载并预处理测试数据 (来自 {data_path})...")
    data = pd.read_csv(data_path)
    processed_data = preprocess_data(data)
    
    # 使用 reindex 保证测试数据的列与训练时完全一致
    X_test = processed_data.reindex(columns=feature_names, fill_value=0)

    print("计算SHAP值...")
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    # 为了在图上显示更清晰的标签，拷贝一份数据并重命名列
    X_test_display = X_test.copy()
    X_test_display.columns = [col.replace('_', ' ') for col in X_test.columns]
    shap_values.feature_names = [name.replace('_', ' ') for name in feature_names]

    print("生成SHAP摘要图（蜂窝图）...")
    plt.figure()
    shap.summary_plot(shap_values, X_test_display, plot_type="dot", show=False, max_display=len(feature_names))
    
    fig = plt.gcf()
    fig.tight_layout()
    
    print(f"保存图片到 {output_path}...")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("SHAP图像生成完成！")

if __name__ == "__main__":
    MODEL_FILE = "hypotension_model.json"
    DATA_FILE = "test.csv"  # 使用测试集来生成SHAP图
    FEATURES_FILE = "model_features.pkl"
    OUTPUT_IMAGE_FILE = "shap.png"
    
    generate_shap_summary_plot(MODEL_FILE, DATA_FILE, FEATURES_FILE, OUTPUT_IMAGE_FILE) 