# Kaggle 房价预测中的集成技巧：实战与指南

## 一、场景描述

在房地产领域，准确预测房价对于投资者、开发商和购房者都具有重要意义。本文的任务是基于 `Kaggle` 竞赛数据集（[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)）中的房屋特征来预测房价。这些特征包括但不限于面积、位置、建造年份、房间数量、车库容量等。数据集中既包含类别型特征（如房屋风格、地理位置等），也包含数值型特征（如面积、价格等）。然而，数据存在缺失值和噪声（如建造年份缺失、测量误差等），需通过预处理构建可靠预测模型。例如，某些房屋的建造年份可能未被记录，或者某些特征的值可能因测量误差而存在偏差。因此，在构建预测模型之前，需要对数据进行预处理，以确保模型能够从清洁和完整的数据中学习。

## 二、集成方案设计

### 基模型选择

为了构建一个强大的集成模型，我们选择了三种在机器学习领域广泛应用且各具优势的基模型：

1. **LightGBM**：
   `LightGBM` 是一种高效的梯度提升框架，特别适用于处理类别型特征。它通过优化的决策树结构和直方图算法，在处理大规模数据时效率显著。LightGBM 能够直接处理类别型特征（需将特征标记为 `category` 类型），无需进行额外的编码转换，但在本文中为统一预处理流程，仍对类别型特征进行独热编码。实际应用中可直接利用 LightGBM 的原生类别型特征优化功能。

2. **XGBoost**：
   `XGBoost` 是另一种著名的梯度提升算法，以其鲁棒性和高准确性而闻名。它在处理数值型特征时具有优势，通过正则化技术有效防止过拟合，同时提供了丰富的参数调整选项。XGBoost 的鲁棒性使其在面对复杂数据分布时仍能保持稳定表现。

3. **随机森林**：
   随机森林是一种基于集成学习的算法，通过构建多个决策树并将其预测结果进行平均来提高模型的准确性和稳定性。它特别擅长捕捉数据中的非线性关系，即使在特征之间存在复杂的交互作用时也能表现出色。随机森林对异常值和噪声具有一定的鲁棒性，这使得它在数据质量不完美的情况下仍能提供可靠的预测。

### 融合方式

1. **加权平均**：
   加权平均是一种简单而有效的集成融合方法。我们根据每个基模型的特点和性能，为其分配不同的权重。具体来说，`LightGBM` 因其在处理类别型特征方面的优势被赋予了较高的权重 **0.5**，`XGBoost` 凭借其鲁棒性和准确性获得权重 **0.3**，而随机森林则因捕捉非线性关系的能力获得权重 **0.2**。通过这种方式，我们可以综合各个模型的优势，同时根据它们的相对重要性进行调整，以期获得更准确的预测结果。

2. **Stacking**：
   `Stacking` 是一种更为高级的集成融合技术。其基本思想是将基模型的预测结果作为新的特征（元特征），然后训练一个元模型来对这些元特征进行学习和整合，从而做出最终的预测。在本文的方案中，我们使用 `LightGBM`、`XGBoost` 和随机森林的预测结果作为元特征，选择岭回归作为元模型。岭回归通过引入 `L2` 正则化项，能够有效地处理元特征之间的多重共线性问题，提高模型的泛化能力。
   
   **关键改进**：为避免数据泄露，我们使用 `K` 折交叉验证生成元特征。具体流程如下：
   
   - 将训练集划分为 **5** 折；
   - 每次使用 **4** 折训练基模型，预测剩余 **1** 折生成元特征；
   - 最终整合所有折的预测结果作为完整的元特征。

## 三、代码实现

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 加载数据
def load_data(train_path, test_path):
    """
    加载训练数据和测试数据
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

# 数据预处理
def preprocess_data(train_data):
    """
    数据预处理，包括特征和目标变量的分割、对数变换等
    """
    # 分割特征和目标变量
    X = train_data.drop('SalePrice', axis=1)
    y = np.log1p(train_data['SalePrice'])  # 对数变换解决价格偏态分布
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

# 特征工程
def feature_engineering(X_train):
    """
    特征工程，包括特征类型识别和预处理管道构建
    """
    # 特征类型识别
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    # 预处理管道
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # 中位数填充缺失值
        ('scaler', StandardScaler())                   # 标准化
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # 填充缺失值
        ('onehot', OneHotEncoder(handle_unknown='ignore'))                     # 独热编码
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

# 基模型定义
def define_base_models():
    """
    定义基模型
    """
    lgb_model = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=200)
    xgb_model = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=200)
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    return lgb_model, xgb_model, rf_model

# 训练管道构建
def build_training_pipelines(preprocessor, base_models):
    """
    构建基模型的训练管道
    """
    lgb_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', base_models[0])])
    xgb_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', base_models[1])])
    rf_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', base_models[2])])
    return lgb_pipe, xgb_pipe, rf_pipe

# 交叉验证生成元特征
def generate_meta_features(X_train, y_train, base_pipes):
    """
    使用交叉验证生成元特征，避免数据泄露
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_features = np.zeros((X_train.shape[0], 3))  # 存储三个基模型的预测结果

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]

        # 训练基模型
        for i, pipe in enumerate(base_pipes):
            pipe.fit(X_train_fold, y_train_fold)
            meta_features[val_idx, i] = pipe.predict(X_val_fold)

    return meta_features

# 元模型训练
def train_meta_model(meta_features, y_train):
    """
    训练元模型
    """
    # 标准化元特征
    scaler = StandardScaler()
    meta_features_scaled = scaler.fit_transform(meta_features)
    
    # 训练元模型
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_features_scaled, y_train)
    
    return meta_model, scaler

# 验证集预测
def make_predictions(base_pipes, meta_model, scaler, X_val):
    """
    在验证集上进行预测
    """
    # 获取基模型的预测值
    lgb_pred = base_pipes[0].predict(X_val)
    xgb_pred = base_pipes[1].predict(X_val)
    rf_pred = base_pipes[2].predict(X_val)
    
    # 打印基模型的预测值（对数变换后）
    # print("LightGBM Predictions (log scale):", lgb_pred)
    # print("XGBoost Predictions (log scale):", xgb_pred)
    # print("Random Forest Predictions (log scale):", rf_pred)
    
    # 构造元特征
    test_meta = np.column_stack([lgb_pred, xgb_pred, rf_pred])
    
    # 标准化元特征
    test_meta_scaled = scaler.transform(test_meta)
    
    # 打印标准化后的元特征
    print("Scaled Meta Features:", test_meta_scaled)
    
    # 元模型预测
    ensemble_pred = meta_model.predict(test_meta_scaled)
    
    # 打印元模型的预测值（对数变换后）
    #print("Ensemble Predictions (log scale):", ensemble_pred)
    
    # 逆转对数变换
    final_pred = np.expm1(ensemble_pred)
    
    # 打印最终预测值（原始 scale）
    # print("Final Predictions (original scale):", final_pred)
    
    return final_pred

# 模型评估
def evaluate_models(y_val, final_pred, base_pipes, X_val):
    """
    统一在原始房价尺度下评估所有模型
    """
    # 真实房价（原始尺度）
    y_val_original = np.expm1(y_val)
    
    # 获取基模型的预测值（log尺度）
    lgb_pred_log = base_pipes[0].predict(X_val)
    xgb_pred_log = base_pipes[1].predict(X_val)
    rf_pred_log = base_pipes[2].predict(X_val)
    
    # 将基模型预测值转换回原始尺度
    lgb_pred_original = np.expm1(lgb_pred_log)
    xgb_pred_original = np.expm1(xgb_pred_log)
    rf_pred_original = np.expm1(rf_pred_log)
    
    # 计算基模型的原始尺度指标
    lgb_rmse = np.sqrt(mean_squared_error(y_val_original, lgb_pred_original))
    xgb_rmse = np.sqrt(mean_squared_error(y_val_original, xgb_pred_original))
    rf_rmse = np.sqrt(mean_squared_error(y_val_original, rf_pred_original))
    
    lgb_r2 = r2_score(y_val_original, lgb_pred_original)
    xgb_r2 = r2_score(y_val_original, xgb_pred_original)
    rf_r2 = r2_score(y_val_original, rf_pred_original)
    
    # 计算集成模型的原始尺度指标
    ensemble_rmse = np.sqrt(mean_squared_error(y_val_original, final_pred))
    ensemble_r2 = r2_score(y_val_original, final_pred)
    
    # 打印统一尺度的评估结果
    print("=== 原始房价尺度评估 ===")
    print(f"LightGBM RMSE: {lgb_rmse:.5f}")
    print(f"XGBoost RMSE:  {xgb_rmse:.5f}")
    print(f"Random Forest RMSE: {rf_rmse:.5f}")
    print(f"Ensemble RMSE: {ensemble_rmse:.5f}\n")
    
    print(f"LightGBM R²: {lgb_r2:.5f}")
    print(f"XGBoost R²:  {xgb_r2:.5f}")
    print(f"Random Forest R²: {rf_r2:.5f}")
    print(f"Ensemble R²: {ensemble_r2:.5f}")

# 可视化
def visualize_results(y_val, final_pred):
    """
    绘制实际值与预测值的散点图
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, final_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted House Prices')
    plt.show()

def plot_learning_curve(estimator, title, X, y):
    """
    绘制学习曲线
    """
    from sklearn.model_selection import learning_curve
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()

# 主函数
if __name__ == "__main__":
    # 加载数据
    train_data, test_data = load_data('train.csv', 'test.csv')

    # 数据预处理
    X_train, X_val, y_train, y_val = preprocess_data(train_data)

    # 特征工程
    preprocessor = feature_engineering(X_train)

    # 定义基模型
    base_models = define_base_models()

    # 构建训练管道
    base_pipes = build_training_pipelines(preprocessor, base_models)

    # 交叉验证生成元特征
    meta_features = generate_meta_features(X_train, y_train, base_pipes)

    # 训练元模型
    meta_model = train_meta_model(meta_features, y_train)

    # 验证集预测
    final_pred = make_predictions(base_pipes, meta_model, X_val)

    # 模型评估
    evaluate_models(y_val, final_pred, base_pipes, X_val)

    # 可视化
    visualize_results(y_val, final_pred)
    plot_learning_curve(meta_model, "Ensemble Model Learning Curve", meta_features, y_train)
```

## 四、方案总结

集成模型不仅在预测准确性上优于单一的基模型，还展现了更强的泛化能力和稳定性。具体来说，加权平均方法通过合理分配各基模型的权重，充分利用了每个模型的优势；而 `Stacking `方法则进一步提升了模型的性能，通过元模型对基模型的预测结果进行整合，挖掘出了更多的信息。

在实际应用中，我们可以根据数据的特点和具体需求对集成方案进行调整和优化。例如，可以尝试更多的基模型，如 `CatBoost`、梯度提升树等，或者采用更复杂的融合方法，如多层 `Stacking`、`Blending` 等。此外，超参数调优也是提升模型性能的重要环节，可以使用网格搜索、随机搜索或贝叶斯优化等方法对基模型和元模型的参数进行精细调整。

总之，集成学习作为一种强大的机器学习技术，在解决复杂预测问题时具有明显的优势。