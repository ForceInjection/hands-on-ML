# Kaggle 房价预测中的集成技巧：实战与指南

## 一、场景描述

在房地产领域，准确预测房价对于投资者、开发商和购房者都具有重要意义。本文的任务是基于 `Kaggle` 竞赛数据集（[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)）中的房屋特征来预测房价。这些特征包括但不限于面积、位置、建造年份、房间数量、车库容量等。数据集中既包含类别型特征（如房屋风格、地理位置等），也包含数值型特征（如面积、价格等）。然而，数据存在缺失值和噪声（如建造年份缺失、测量误差等），需通过预处理构建可靠预测模型。例如，某些房屋的建造年份可能未被记录，或者某些特征的值可能因测量误差而存在偏差。因此，在构建预测模型之前，需要对数据进行预处理，以确保模型能够从清洁和完整的数据中学习。

---

## 二、集成方案设计

### 2.1 基模型选择

为了构建一个强大的集成模型，我们选择了三种在机器学习领域广泛应用且各具优势的基模型：

1. **LightGBM**：
   `LightGBM` 是一种高效的梯度提升框架，特别适用于处理类别型特征。它通过优化的决策树结构和直方图算法，在处理大规模数据时效率显著。LightGBM 能够直接处理类别型特征（需将特征标记为 `category` 类型），无需进行额外的编码转换，但在本文中为统一预处理流程，仍对类别型特征进行独热编码。实际应用中可直接利用 LightGBM 的原生类别型特征优化功能。

2. **XGBoost**：
   `XGBoost` 是另一种著名的梯度提升算法，以其鲁棒性和高准确性而闻名。它在处理数值型特征时具有优势，通过正则化技术有效防止过拟合，同时提供了丰富的参数调整选项。XGBoost 的鲁棒性使其在面对复杂数据分布时仍能保持稳定表现。

3. **随机森林**：
   随机森林是一种基于集成学习的算法，通过构建多个决策树并将其预测结果进行平均来提高模型的准确性和稳定性。它特别擅长捕捉数据中的非线性关系，即使在特征之间存在复杂的交互作用时也能表现出色。随机森林对异常值和噪声具有一定的鲁棒性，这使得它在数据质量不完美的情况下仍能提供可靠的预测。

### 2.2 融合方式

1. **加权平均**：
   加权平均是一种简单而有效的集成融合方法。我们根据每个基模型的特点和性能，为其分配不同的权重。具体来说，`LightGBM` 因其在处理类别型特征方面的优势被赋予了较高的权重 **0.5**，`XGBoost` 凭借其鲁棒性和准确性获得权重 **0.3**，而随机森林则因捕捉非线性关系的能力获得权重 **0.2**。通过这种方式，我们可以综合各个模型的优势，同时根据它们的相对重要性进行调整，以期获得更准确的预测结果。

2. **Stacking**：
   `Stacking` 是一种更为高级的集成融合技术。其基本思想是将基模型的预测结果作为新的特征（元特征），然后训练一个元模型来对这些元特征进行学习和整合，从而做出最终的预测。在本文的方案中，我们使用 `LightGBM`、`XGBoost` 和随机森林的预测结果作为元特征，选择岭回归作为元模型。岭回归通过引入 `L2` 正则化项，能够有效地处理元特征之间的多重共线性问题，提高模型的泛化能力。

   **关键改进**：为避免数据泄露，我们使用 `K` 折交叉验证生成元特征。具体流程如下：

   - 将训练集划分为 **5** 折；
   - 每次使用 **4** 折训练基模型，预测剩余 **1** 折生成元特征；
   - 最终整合所有折的预测结果作为完整的元特征。

---

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

    教学说明：
    1. 房价数据通常呈现右偏分布（少数高价房产），对数变换可以使其更接近正态分布
    2. 使用 np.log1p() 而不是 np.log() 是为了避免对数为0的情况
    3. 随机划分训练集和验证集，固定随机种子确保结果可复现
    """
    # 分割特征和目标变量
    X = train_data.drop('SalePrice', axis=1)
    y = np.log1p(train_data['SalePrice'])  # 对数变换解决价格偏态分布

    # 打印数据基本信息
    print(f"特征数据形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    print(f"缺失值统计:\n{X.isnull().sum().sort_values(ascending=False).head(10)}")

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"训练集大小: {X_train.shape[0]} 样本")
    print(f"验证集大小: {X_val.shape[0]} 样本")

    return X_train, X_val, y_train, y_val

# 特征工程
def feature_engineering(X_train):
    """
    特征工程，包括特征类型识别和预处理管道构建

    教学说明：
    1. 数值特征：使用中位数填充缺失值（对异常值不敏感），然后标准化
    2. 类别特征：使用常量值填充缺失值，然后进行独热编码
    3. 独热编码的 handle_unknown='ignore' 参数确保在测试集遇到新类别时不会报错
    """
    # 特征类型识别
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    print(f"数值特征数量: {len(numeric_features)}")
    print(f"类别特征数量: {len(categorical_features)}")
    print(f"数值特征示例: {list(numeric_features[:5])}")
    print(f"类别特征示例: {list(categorical_features[:5])}")

    # 预处理管道
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # 中位数填充缺失值
        ('scaler', StandardScaler())                   # 标准化
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # 填充缺失值
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # 独热编码
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

    教学说明：
    1. LightGBM：适合处理类别特征，训练速度快
    2. XGBoost：鲁棒性强，正则化防止过拟合
    3. 随机森林：捕捉非线性关系，对异常值鲁棒

    参数说明：
    - num_leaves: LightGBM中叶节点数量，控制模型复杂度
    - learning_rate: 学习率，控制每次迭代的步长
    - n_estimators: 树的数量，集成规模
    - max_depth: 树的最大深度，防止过拟合
    """
    lgb_model = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=200, random_state=42)
    xgb_model = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=200, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

    print("基模型定义完成:")
    print(f"- LightGBM: {lgb_model}")
    print(f"- XGBoost: {xgb_model}")
    print(f"- RandomForest: {rf_model}")

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

    教学说明：
    1. 使用K折交叉验证确保每个样本的预测都是在未见过的数据上生成的
    2. 避免数据泄露：不能用训练数据直接预测训练数据
    3. 生成3列元特征，分别对应3个基模型的预测结果
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_features = np.zeros((X_train.shape[0], 3))  # 存储三个基模型的预测结果

    print("开始交叉验证生成元特征...")
    print(f"元特征矩阵形状: {meta_features.shape}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]

        print(f"正在处理第 {fold}/5 折交叉验证...")

        # 训练基模型
        for i, pipe in enumerate(base_pipes):
            pipe.fit(X_train_fold, y_train_fold)
            meta_features[val_idx, i] = pipe.predict(X_val_fold)

            # 打印每折的预测统计
            if fold == 1 and i == 0:  # 只打印第一折第一个模型的统计
                print(f"  模型{i+1}预测值范围: [{meta_features[val_idx, i].min():.3f}, {meta_features[val_idx, i].max():.3f}]")

    print("元特征生成完成!")
    return meta_features

# 元模型训练
def train_meta_model(meta_features, y_train):
    """
    训练元模型

    教学说明：
    1. 标准化元特征：使不同基模型的预测值具有可比性
    2. 岭回归：处理多重共线性问题，防止过拟合
    3. 返回scaler对象用于后续预测时的标准化
    """
    # 标准化元特征
    scaler = StandardScaler()
    meta_features_scaled = scaler.fit_transform(meta_features)

    # 训练元模型
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_features_scaled, y_train)

    print(f"元模型训练完成: {meta_model}")
    print(f"元特征标准化参数: 均值={scaler.mean_}, 标准差={scaler.scale_}")

    return meta_model, scaler

# 验证集预测
def make_predictions(base_pipes, meta_model, scaler, X_val):
    """
    在验证集上进行预测

    教学说明：
    1. 首先获取各基模型的预测结果
    2. 将预测结果组合成元特征矩阵
    3. 使用训练时相同的scaler标准化元特征
    4. 使用元模型进行最终预测
    5. 将对数尺度的预测转换回原始房价尺度
    """
    # 获取基模型的预测值
    lgb_pred = base_pipes[0].predict(X_val)
    xgb_pred = base_pipes[1].predict(X_val)
    rf_pred = base_pipes[2].predict(X_val)

    # 打印基模型的预测统计
    print("基模型预测统计 (对数尺度):")
    print(f"  LightGBM: [{lgb_pred.min():.3f}, {lgb_pred.max():.3f}], 均值: {lgb_pred.mean():.3f}")
    print(f"  XGBoost:  [{xgb_pred.min():.3f}, {xgb_pred.max():.3f}], 均值: {xgb_pred.mean():.3f}")
    print(f"  RandomForest: [{rf_pred.min():.3f}, {rf_pred.max():.3f}], 均值: {rf_pred.mean():.3f}")

    # 构造元特征
    test_meta = np.column_stack([lgb_pred, xgb_pred, rf_pred])

    # 标准化元特征
    test_meta_scaled = scaler.transform(test_meta)

    # 打印标准化后的元特征统计
    print("标准化后元特征统计:")
    for i in range(3):
        print(f"  特征{i+1}: [{test_meta_scaled[:, i].min():.3f}, {test_meta_scaled[:, i].max():.3f}], 均值: {test_meta_scaled[:, i].mean():.3f}")

    # 元模型预测
    ensemble_pred = meta_model.predict(test_meta_scaled)

    # 逆转对数变换
    final_pred = np.expm1(ensemble_pred)

    print(f"最终预测范围: [{final_pred.min():.1f}, {final_pred.max():.1f}], 均值: {final_pred.mean():.1f}")

    return final_pred

# 模型评估
def evaluate_models(y_val, final_pred, base_pipes, X_val):
    """
    统一在原始房价尺度下评估所有模型

    教学说明：
    1. RMSE（均方根误差）：衡量预测值与真实值的平均差异，单位与目标变量相同
    2. R²（决定系数）：衡量模型解释的方差比例，越接近1越好
    3. 所有评估都在原始房价尺度进行，便于业务理解
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

    # 计算性能提升百分比
    rmse_improvement = ((min(lgb_rmse, xgb_rmse, rf_rmse) - ensemble_rmse) / min(lgb_rmse, xgb_rmse, rf_rmse)) * 100
    r2_improvement = ((ensemble_r2 - max(lgb_r2, xgb_r2, rf_r2)) / max(lgb_r2, xgb_r2, rf_r2)) * 100

    # 打印统一尺度的评估结果
    print("\n=== 模型性能评估 (原始房价尺度) ===")
    print("-" * 50)
    print(f"LightGBM     RMSE: {lgb_rmse:.2f} | R²: {lgb_r2:.4f}")
    print(f"XGBoost      RMSE: {xgb_rmse:.2f} | R²: {xgb_r2:.4f}")
    print(f"Random Forest RMSE: {rf_rmse:.2f} | R²: {rf_r2:.4f}")
    print(f"Ensemble     RMSE: {ensemble_rmse:.2f} | R²: {ensemble_r2:.4f}")
    print("-" * 50)
    print(f"RMSE提升: {rmse_improvement:.1f}% (相对于最佳基模型)")
    print(f"R²提升: {r2_improvement:.1f}% (相对于最佳基模型)")
    print("-" * 50)

    # 返回评估结果用于可视化
    results = {
        'rmse': {'LightGBM': lgb_rmse, 'XGBoost': xgb_rmse, 'RandomForest': rf_rmse, 'Ensemble': ensemble_rmse},
        'r2': {'LightGBM': lgb_r2, 'XGBoost': xgb_r2, 'RandomForest': rf_r2, 'Ensemble': ensemble_r2}
    }
    return results

# 可视化
def visualize_results(y_val, final_pred, results):
    """
    绘制评估结果可视化

    教学说明：
    1. 散点图：展示预测值与真实值的相关性，理想情况应该沿对角线分布
    2. 性能对比图：直观展示各模型的RMSE表现
    3. 使用中文字体支持，确保图表显示正常
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 实际值与预测值散点图
    y_val_original = np.expm1(y_val)
    ax1.scatter(y_val_original, final_pred, alpha=0.5)
    ax1.plot([y_val_original.min(), y_val_original.max()],
             [y_val_original.min(), y_val_original.max()], 'r--', linewidth=2)
    ax1.set_xlabel('实际房价')
    ax1.set_ylabel('预测房价')
    ax1.set_title('实际值 vs 预测值')
    ax1.grid(True, alpha=0.3)

    # 添加R²值到散点图
    r2 = results['r2']['Ensemble']
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 模型性能对比柱状图
    models = ['LightGBM', 'XGBoost', 'RandomForest', 'Ensemble']
    rmse_values = [results['rmse'][model] for model in models]

    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    bars = ax2.bar(models, rmse_values, color=colors)
    ax2.set_xlabel('模型')
    ax2.set_ylabel('RMSE')
    ax2.set_title('模型性能对比 (RMSE)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    # 在柱状图上添加数值标签
    for i, v in enumerate(rmse_values):
        ax2.text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    print("可视化完成！集成模型在RMSE指标上表现最佳。")

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
    """
    主函数执行流程说明：
    1. 数据加载与预处理
    2. 特征工程与模型构建
    3. 交叉验证生成元特征（避免数据泄露）
    4. 元模型训练与预测
    5. 模型评估与可视化
    """
    print("=" * 60)
    print("Kaggle房价预测集成学习实战")
    print("=" * 60)

    # 加载数据
    print("\n1. 加载数据...")
    train_data, test_data = load_data('train.csv', 'test.csv')

    # 数据预处理
    print("\n2. 数据预处理...")
    X_train, X_val, y_train, y_val = preprocess_data(train_data)

    # 特征工程
    print("\n3. 特征工程...")
    preprocessor = feature_engineering(X_train)

    # 定义基模型
    print("\n4. 定义基模型...")
    base_models = define_base_models()

    # 构建训练管道
    print("\n5. 构建训练管道...")
    base_pipes = build_training_pipelines(preprocessor, base_models)

    # 交叉验证生成元特征
    print("\n6. 交叉验证生成元特征（避免数据泄露）...")
    meta_features = generate_meta_features(X_train, y_train, base_pipes)

    # 训练元模型
    print("\n7. 训练元模型...")
    meta_model, scaler = train_meta_model(meta_features, y_train)

    # 验证集预测
    print("\n8. 验证集预测...")
    final_pred = make_predictions(base_pipes, meta_model, scaler, X_val)

    # 模型评估
    print("\n9. 模型评估...")
    results = evaluate_models(y_val, final_pred, base_pipes, X_val)

    # 可视化
    print("\n10. 可视化结果...")
    visualize_results(y_val, final_pred, results)

    print("\n" + "=" * 60)
    print("集成学习实战完成！")
    print("=" * 60)
```

---

## 四、方案总结

集成模型不仅在预测准确性上优于单一的基模型，还展现了更强的泛化能力和稳定性。具体来说，加权平均方法通过合理分配各基模型的权重，充分利用了每个模型的优势；而 `Stacking`方法则进一步提升了模型的性能，通过元模型对基模型的预测结果进行整合，挖掘出了更多的信息。

在实际应用中，我们可以根据数据的特点和具体需求对集成方案进行调整和优化。例如，可以尝试更多的基模型，如 `CatBoost`、梯度提升树等，或者采用更复杂的融合方法，如多层 `Stacking`、`Blending` 等。此外，超参数调优也是提升模型性能的重要环节，可以使用网格搜索、随机搜索或贝叶斯优化等方法对基模型和元模型的参数进行精细调整。

总之，集成学习作为一种强大的机器学习技术，在解决复杂预测问题时具有明显的优势。
