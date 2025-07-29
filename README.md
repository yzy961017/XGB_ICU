# XGBoost ICU Hypotension Prediction Model

这是一个使用XGBoost预测ICU患者低血压的机器学习项目。

## 项目依赖管理

本项目使用 [uv](https://docs.astral.sh/uv/) 进行Python依赖管理。uv是一个快速的Python包管理器，提供了类似于npm或cargo的现代化依赖管理体验。

### 安装uv

如果您还没有安装uv，请访问 [uv官方文档](https://docs.astral.sh/uv/getting-started/installation/) 获取安装说明。

### 项目设置

1. **克隆项目后，安装依赖：**
   ```bash
   uv sync
   ```

2. **激活虚拟环境：**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **或者直接使用uv运行命令：**
   ```bash
   uv run python app.py
   uv run streamlit run app.py
   ```

### 依赖管理

- **添加新依赖：**
  ```bash
  uv add package-name
  ```

- **添加开发依赖：**
  ```bash
  uv add --dev package-name
  ```

- **移除依赖：**
  ```bash
  uv remove package-name
  ```

- **更新依赖：**
  ```bash
  uv lock --upgrade
  ```

### 项目文件说明

- `pyproject.toml` - 项目配置和依赖声明
- `uv.lock` - 锁定的依赖版本（确保可重现的构建）
- `requirements.txt` - 传统的pip依赖文件（保留用于兼容性）

### 主要依赖

- **scikit-learn** - 机器学习库
- **xgboost** - XGBoost梯度提升库
- **pandas** - 数据处理
- **numpy** - 数值计算
- **matplotlib** - 数据可视化
- **plotly** - 交互式图表
- **streamlit** - Web应用框架
- **shap** - 模型解释性
- **lime** - 局部可解释性
- **Pillow** - 图像处理

### 运行应用

使用uv运行Streamlit应用：
```bash
uv run streamlit run app.py
```

### 优势

使用uv相比传统的pip + virtualenv有以下优势：

1. **更快的依赖解析和安装**
2. **自动虚拟环境管理**
3. **锁定文件确保可重现的构建**
4. **现代化的依赖管理体验**
5. **与现有Python生态系统完全兼容**

### 传统方式（备用）

如果您更喜欢使用传统的pip方式：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
