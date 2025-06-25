import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def preprocess_data(data: pd.DataFrame):
    """对数据进行预处理，包括重命名、二元特征转换和独热编码"""
    
    # 移除CSV中多余的索引列
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    
    # --- 标准化特征名称 ---
    if 'vasoactive.drugs' in data.columns:
        data = data.rename(columns={'vasoactive.drugs': 'vasoactive_drugs'})
    
    # --- 数据预处理 ---
    # 1. 将所有"是"/"否"二元特征转换为 1/0
    binary_map = {'是': 1, '否': 0}
    binary_cols = [
        'congestive_heart_failure', 'peripheral_vascular_disease', 'dementia', 
        'chronic_pulmonary_disease', 'mild_liver_disease', 'diabetes_without_cc', 
        'malignant_cancer', 'metastatic_solid_tumor', 'vasoactive_drugs'
    ]
    for col in binary_cols:
        if col in data.columns and data[col].dtype == 'object':
            data[col] = data[col].map(binary_map)

    # 2. 对多分类变量进行独热编码
    categorical_cols = ['gender', 'rrt_type']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True, dtype=int)
    
    return data

def train_and_save_model(train_data_path: str, test_data_path: str, model_output_path: str):
    """
    加载数据、训练XGBoost模型并保存。
    
    Args:
        train_data_path (str): 训练数据CSV文件的路径。
        test_data_path (str): 测试数据CSV文件的路径。
        model_output_path (str): 保存训练好模型的文件路径。
    """
    print("开始加载并处理训练数据...")
    try:
        train_data = pd.read_csv(train_data_path, index_col=False, header=0, encoding='utf-8')
        train_data = preprocess_data(train_data)
        
        # 确保有目标列
        if 'hypotension' not in train_data.columns:
            raise ValueError("训练数据中缺少目标列 'hypotension'")
        
        print(f"训练数据加载成功，共 {len(train_data)} 条记录。")
    except Exception as e:
        print(f"训练数据处理出错: {e}")
        return

    print("开始加载并处理测试数据...")
    try:
        test_data = pd.read_csv(test_data_path, index_col=False, header=0, encoding='utf-8')
        test_data = preprocess_data(test_data)
        
        # 确保有目标列
        if 'hypotension' not in test_data.columns:
            raise ValueError("测试数据中缺少目标列 'hypotension'")
        
        print(f"测试数据加载成功，共 {len(test_data)} 条记录。")
    except Exception as e:
        print(f"测试数据处理出错: {e}")
        return

    # 构建特征列表
    # 注意：这里的特征列表是基于预处理后的列名
    binary_cols = [
        'congestive_heart_failure', 'peripheral_vascular_disease', 'dementia', 
        'chronic_pulmonary_disease', 'mild_liver_disease', 'diabetes_without_cc', 
        'malignant_cancer', 'metastatic_solid_tumor', 'vasoactive_drugs'
    ]
    feature_cols = [
        'admission_age', 'ph', 'lactate', 'icu_to_rrt_hours', 'map', 'sap'
    ] + binary_cols + [col for col in train_data.columns if 'gender_' in col or 'rrt_type_' in col]
    
    # 确保训练集和测试集的列一致
    # 获取训练数据独热编码后的所有特征列
    train_cols = set(train_data.columns)
    # 获取测试数据独热编码后的所有特征列
    test_cols = set(test_data.columns)

    # 找出只在训练集中存在的列，并添加到测试集中，填充0
    for col in train_cols - test_cols:
        if col != 'hypotension': # 不处理目标列
             test_data[col] = 0

    # 找出只在测试集中存在的列，并添加到训练集中，填充0
    for col in test_cols - train_cols:
        if col != 'hypotension': # 不处理目标列
            train_data[col] = 0
            
    # 按最终的特征列表顺序重新排序列
    test_data = test_data[train_data.columns]
        
    X_train = train_data[feature_cols]
    y_train = train_data["hypotension"]
    
    X_test = test_data[feature_cols]
    y_test = test_data["hypotension"]
    
    print(f"数据准备完成，共 {len(X_train.columns)} 个特征。")

    # XGBoost的超参数空间
    xgb_param_dist = {
        'n_estimators': range(100, 501, 50),
        'max_depth': range(3, 11),
        'learning_rate': np.linspace(0.01, 0.3, 10),
        'subsample': np.linspace(0.7, 1.0, 4),
        'colsample_bytree': np.linspace(0.7, 1.0, 4),
        'min_child_weight': range(1, 6),
        'gamma': np.linspace(0, 0.5, 6)
    }

    print("开始进行超参数搜索...")
    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(
        model,
        param_distributions=xgb_param_dist,
        n_iter=50,
        scoring="roc_auc",
        n_jobs=-1,
        cv=cv,
        random_state=42,
        verbose=1
    )
    
    # --- 只在训练集上进行搜索 ---
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    print(f"搜索完成。最佳超参数: {best_params}")

    # --- 使用最佳模型在测试集上进行验证 ---
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n--- 模型验证结果 (在测试集上) ---")
    print(f"AUC 分数: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    print("-------------------------------------\n")

    print("验证完成。使用最佳参数在【完整训练集】上训练最终模型...")
    final_xgb_model = xgb.XGBClassifier(
        random_state=42, 
        eval_metric='logloss',
        **best_params
    )
    final_xgb_model.fit(X_train, y_train) # 使用全部训练数据进行最终训练
    print("模型训练完成。")

    print(f"正在将模型保存到 {model_output_path}...")
    try:
        final_xgb_model.save_model(model_output_path)
        print("模型保存成功。")
        
        # 保存特征列表
        feature_list_path = "model_features.pkl"
        with open(feature_list_path, 'wb') as f:
            pickle.dump(feature_cols, f)
        print(f"特征列表保存到 {feature_list_path}")
        
    except Exception as e:
        print(f"模型保存失败: {e}")

if __name__ == "__main__":
    TRAIN_FILE = "./train.csv"
    TEST_FILE = "./test.csv"
    MODEL_FILE = "hypotension_model.json"
    train_and_save_model(TRAIN_FILE, TEST_FILE, MODEL_FILE)