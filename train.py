import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def preprocess_data(data: pd.DataFrame):
    """对数据进行预处理，包括重命名、二元特征转换和独热编码"""
    
    # 移除CSV中多余的索引列
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
    
    # --- 标准化特征名称 ---
    # 处理vasoactive.drugs列名中的点号
    if 'vasoactive.drugs' in data.columns:
        data = data.rename(columns={'vasoactive.drugs': 'vasoactive_drugs'})
    
    # --- 数据预处理 ---
    # 1. 将性别转换为数值 (M=1, F=0)
    if 'gender' in data.columns:
        data['gender'] = data['gender'].map({'M': 1, 'F': 0})
    
    # 2. 对RRT类型进行独热编码
    if 'rrt_type' in data.columns:
        data = pd.get_dummies(data, columns=['rrt_type'], prefix='rrt_type', drop_first=True, dtype=int)
    
    return data

def train_and_save_model(train_data_path: str, test_data_path: str, model_output_path: str):
    """
    加载数据、训练GBM模型并保存。
    
    Args:
        train_data_path (str): 训练数据CSV文件的路径。
        test_data_path (str): 测试数据CSV文件的路径。
        model_output_path (str): 保存训练好模型的文件路径。
    """
    print("开始加载并处理训练数据...")
    try:
        train_data = pd.read_csv(train_data_path, index_col=False, header=0, encoding='utf-8')
        train_data = preprocess_data(train_data)
        
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
        
        if 'hypotension' not in test_data.columns:
            raise ValueError("测试数据中缺少目标列 'hypotension'")
        
        print(f"测试数据加载成功，共 {len(test_data)} 条记录。")
    except Exception as e:
        print(f"测试数据处理出错: {e}")
        return

    # 对齐训练集和测试集的列
    train_cols = set(train_data.columns)
    test_cols = set(test_data.columns)

    # 找出只在训练集中存在的列，并添加到测试集中，填充0
    missing_in_test = train_cols - test_cols
    if 'hypotension' in missing_in_test:
        missing_in_test.remove('hypotension')
    for col in missing_in_test:
        test_data[col] = 0
        print(f"测试集中缺少列 '{col}'，已填充为0")

    # 找出只在测试集中存在的列，并添加到训练集中，填充0
    missing_in_train = test_cols - train_cols
    if 'hypotension' in missing_in_train:
        missing_in_train.remove('hypotension')
    for col in missing_in_train:
        train_data[col] = 0
        print(f"训练集中缺少列 '{col}'，已填充为0")
            
    # 保证测试集和训练集的列顺序完全一致
    test_data = test_data[train_data.columns]

    # 根据实际数据构建特征列表
    feature_cols = [
        'gender',                    # Gender
        'admission_age',             # Age
        'congestive_heart_failure',  # Congestive heart failure
        'peripheral_vascular_disease', # Peripheral vascular disease
        'dementia',                  # Dementia
        'chronic_pulmonary_disease', # Chronic pulmonary disease
        'mild_liver_disease',        # Liver disease
        'diabetes_without_cc',       # Diabetes
        'malignant_cancer',          # Cancer
        'vasoactive_drugs',          # Vasoactive drugs
        'ph',                        # PH
        'lactate',                   # Lactate
        'map',                       # MAP
        'sap',                       # SAP
        'icu_to_rrt_hours'           # ICU to RRT initiation
    ]
    
    # 添加RRT类型的独热编码列（如果存在）
    rrt_cols = [col for col in train_data.columns if col.startswith('rrt_type_')]
    feature_cols.extend(rrt_cols)
    
    # 确保所有特征列都存在
    missing_features = []
    for col in feature_cols:
        if col not in train_data.columns:
            missing_features.append(col)
    
    if missing_features:
        print(f"警告：以下特征列在数据中不存在: {missing_features}")
        # 移除不存在的特征
        feature_cols = [col for col in feature_cols if col in train_data.columns]
    
    print(f"最终使用的特征列: {feature_cols}")

    X_train = train_data[feature_cols]
    y_train = train_data["hypotension"]
    
    X_test = test_data[feature_cols]
    y_test = test_data["hypotension"]
    
    print(f"数据准备完成，共 {len(X_train.columns)} 个特征。")
    print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
    print(f"训练集正例比例: {y_train.mean():.3f}, 测试集正例比例: {y_test.mean():.3f}")

    # GBM的超参数空间
    gbm_param_dist = {
        'n_estimators': range(50, 201, 50),
        'max_depth': range(3, 8),
        'learning_rate': np.linspace(0.01, 0.2, 5),
        'subsample': np.linspace(0.7, 1.0, 4),
        'min_samples_leaf': range(20, 51, 10),
        'min_samples_split': range(100, 201, 50)
    }

    print("开始进行超参数搜索...")
    model = GradientBoostingClassifier(random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(
        model,
        param_distributions=gbm_param_dist,
        n_iter=30,
        scoring="roc_auc",
        n_jobs=-1,
        cv=cv,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    print(f"搜索完成。最佳超参数: {best_params}")

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n--- 模型验证结果 (在测试集上) ---")
    print(f"AUC 分数: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    print("-------------------------------------\n")

    print("验证完成。使用最佳参数在【完整训练集】上训练最终模型...")
    final_gbm_model = GradientBoostingClassifier(
        random_state=42, 
        **best_params
    )
    final_gbm_model.fit(X_train, y_train)
    print("模型训练完成。")

    print(f"正在将模型保存到 hypotension_model.pkl ...")
    try:
        import pickle
        with open("hypotension_model.pkl", "wb") as f:
            pickle.dump(final_gbm_model, f)
        print("模型以pkl格式保存成功。")
        feature_list_path = "model_features.pkl"
        import joblib
        joblib.dump(feature_cols, feature_list_path)
        print(f"特征列表保存到 {feature_list_path}")
    except Exception as e:
        print(f"模型或特征列表保存失败: {e}")

if __name__ == '__main__':
    # 定义文件路径
    TRAIN_FILE = 'train.csv'
    TEST_FILE = 'test.csv'
    MODEL_FILE = 'hypotension_model.joblib'
    
    # 执行训练流程
    train_and_save_model(TRAIN_FILE, TEST_FILE, MODEL_FILE)