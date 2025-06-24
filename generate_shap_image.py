import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import pickle

def generate_shap_summary_plot(model_path, data_path, feature_list_path, output_path):
    """
    加载模型和数据，生成并保存SHAP蜂群图。
    """
    # 配置中文字体，以防标签显示为方框
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        print("警告: 未找到中文字体'SimHei'，图表可能无法正确显示中文。")

    print("加载模型...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    print("加载特征列表...")
    with open(feature_list_path, 'rb') as f:
        feature_names = pickle.load(f)

    print("加载并预处理训练数据...")
    data = pd.read_csv(data_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)

    # --- 标准化特征名称 (与 train.py 一致) ---
    if 'vasoactive.drugs' in data.columns:
        data = data.rename(columns={'vasoactive.drugs': 'vasoactive_drugs'})

    # --- 数据预处理 (与 train.py 完全一致) ---
    # 1. 将所有"是"/"否"二元特征转换为 1/0
    binary_map = {'是': 1, '否': 0}
    binary_cols = [
        'congestive_heart_failure', 'peripheral_vascular_disease', 'dementia', 
        'chronic_pulmonary_disease', 'mild_liver_disease', 'diabetes_without_cc', 
        'malignant_cancer', 'metastatic_solid_tumor', 'vasoactive_drugs'
    ]
    for col in binary_cols:
        if col in data.columns:
            # 确保只对object类型的列进行map，避免对已经转换的列操作
            if data[col].dtype == 'object':
                data[col] = data[col].map(binary_map)

    # 2. 对多分类变量进行独热编码
    categorical_cols = ['gender', 'rrt_type']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # 3. 使用 reindex 保证特征列完全对齐
    X_train = data.reindex(columns=feature_names, fill_value=0)

    # 4. (关键修复) 确保所有列都是纯数字类型，将bool转换为int
    for col in X_train.columns:
        if X_train[col].dtype == 'bool':
            X_train[col] = X_train[col].astype(int)

    print("计算SHAP值 (使用新的Explainer API)...")
    # 使用更新、更稳健的 Explainer API
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    print("生成SHAP条形图...")
    plt.clf() # 清除任何现有的图形
    
    # 新的 Explainer API 返回一个 Explanation 对象，可以直接传递给绘图函数
    # 改为生成条形图，因为它更稳健
    shap.summary_plot(shap_values, plot_type="bar", show=False)
    
    fig = plt.gcf()
    fig.tight_layout() # 调整布局防止标签被截断
    
    print(f"保存图片到 {output_path}...")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("完成！")

if __name__ == "__main__":
    MODEL_FILE = "hypotension_model.json"
    DATA_FILE = "train.csv"
    FEATURES_FILE = "model_features.pkl"
    OUTPUT_IMAGE_FILE = "shap.png"
    
    generate_shap_summary_plot(MODEL_FILE, DATA_FILE, FEATURES_FILE, OUTPUT_IMAGE_FILE) 