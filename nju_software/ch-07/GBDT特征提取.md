# GBDT 特征提取

## 1. 引言

在机器学习任务中，特征工程是影响模型性能的关键因素之一。如何从原始数据中提取有意义的特征，不仅直接决定了模型的最终效果，还影响着模型的泛化能力和鲁棒性。`GBDT`（梯度提升决策树）不仅是一种强大的分类和回归算法，还可以用来进行特征提取，为后续模型提供更好的输入数据。树模型通过递归分裂天然具备特征选择和组合的能力，能够自动捕捉高阶非线性关系，这使得 GBDT 在特征工程中具有独特优势。

## 2. GBDT 简介

`GBDT`（`Gradient Boosting Decision Tree`）是一种集成学习方法，通过构建多个弱决策树（通常是 `CART` 树）来提升整体模型的预测能力。相比于随机森林（`Random Forest`），`GBDT` 采用的是 **boosting**（提升）策略，而非 **bagging**（装袋）。`GBDT 的优势包括：

- 强大的非线性建模能力  
- 现代实现（如 LightGBM、XGBoost）支持高效处理大规模数据，能够在分布式环境下运行，适用于处理海量数据集  
- 具备自动特征选择和组合的能力  

常见的 GBDT 实现包括：

- `sklearn.ensemble.GradientBoostingClassifier`，适合中小规模数据，参数调节灵活  
- `XGBoost`，提供高效的并行计算和交叉验证功能，适用于多种数据场景  
- `LightGBM`，采用直方图算法和梯度基于的单边采样，显著降低内存使用并提升训练速度  
- `CatBoost`，对类别特征处理友好，自动支持文本特征，减少预处理工作量  

---

## 3. GBDT 特征提取的原理 
 
`GBDT` 进行特征提取的方式主要包括以下几种：

### 3.1 叶子索引编码（Leaf Index Encoding）  

`GBDT` 训练完成后，每个样本都会依次落入每棵树的叶子节点。将这些叶子节点的索引组合成新特征，可以表征样本在树结构中的路径信息。例如，若模型包含 `100` 棵树，每个样本将生成 `100` 个叶子索引，可通过 One-Hot 编码或嵌入（`Embedding`）转换为高维稀疏特征，从而捕捉样本在非线性空间中的分布特性。

### 3.2 特征重要性计算（Feature Importance）  

GBDT 通过以下方式计算特征重要性：

- **基于分裂次数**：统计特征在树分裂中被使用的总次数（如 LightGBM 默认方式），适用于快速评估特征的总体贡献  
- **基于信息增益**：计算特征分裂带来的信息增益总和（如 XGBoost 的 `gain` 模式），能反映特征对模型预测能力的提升程度  
- **基于纯度提升**：衡量特征分裂对损失函数的优化程度（如 scikit-learn 实现），直接关联特征与模型性能的改善  

### 3.3 SHAP 解释（SHapley Additive exPlanations）  

`SHAP` 值基于博弈论公平分配特征贡献，可解释单个预测的特征影响。相比传统重要性指标，SHAP 能区分正负向影响，支持全局分析和局部解释，帮助理解模型决策的内在逻辑。

## 4. GBDT 特征提取的流程  
### 4.1 训练 GBDT 模型  

使用完整数据集训练 `GBDT` 基准模型，需注意避免过拟合，可通过早停法、调整树深度、控制学习率等策略实现，确保模型在训练集和测试集上均具有良好的泛化能力。

### 4.2 提取叶子索引特征  

将原始特征转换为叶子索引矩阵（样本数 × 树数），需统一处理训练集与测试集的编码，保证两者在相同的特征空间下，避免因编码差异导致模型输入不一致。

### 4.3 计算特征重要性  

结合 `feature_importances_` 和 `SHAP` 值进行交叉验证，优先保留稳定重要的特征，同时分析特征在不同模型和数据子集上的表现，确保特征选择的可靠性和稳定性。

### 4.4 结合后续模型  

高维稀疏的叶子特征适合线性模型（如 LR），也可通过降维输入神经网络。经典组合包括：  

- `GBDT` + `LR`（`Facebook CTR` 预估方案），利用 GBDT 提取的非线性特征增强 LR 的表达能力  
- `GBDT` + `DNN`（`Wide & Deep` 扩展），结合树模型的高阶特征交互和神经网络的自动学习能力  

## 5. Python 代码示例  

以下示例使用 `LightGBM` 进行 `GBDT` 特征提取。

### 5.1 训练 LightGBM 并安全编码叶子索引  

```python
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

# 生成示例数据
X = np.random.rand(1000, 10)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=100, num_leaves=31)
lgb_model.fit(X_train, y_train)

# 提取叶子索引
X_train_leaves = lgb_model.predict(X_train, pred_leaf=True)
X_test_leaves = lgb_model.predict(X_test, pred_leaf=True)

# 使用统一的 OneHot 编码器
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train_leaves)
X_test_encoded = encoder.transform(X_test_leaves)  # 使用相同编码

# 训练带正则化的逻辑回归
lr = LogisticRegression(penalty='l1', solver='liblinear')  # 增加稀疏特征处理
lr.fit(X_train_encoded, y_train)

# 计算准确率
accuracy = lr.score(X_test_encoded, y_test)
print(f"LR 预测准确率: {accuracy:.4f}")
```

### 5.2 计算特征重要性

```python
import matplotlib.pyplot as plt

# 获取特征重要性
feature_importance = lgb_model.feature_importances_

# 可视化
plt.bar(range(len(feature_importance)), feature_importance)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Feature Importance in GBDT")
plt.show()
```
### 5.3 使用 SHAP 进行解释

```python
import shap

# 计算 SHAP 值
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_test)[1]  # 索引1对应正类

# 可视化 SHAP 值
shap.summary_plot(shap_values, X_test)
```


## 6. GBDT 特征提取的应用场景

### 6.1 CTR 预估与推荐系统

`Facebook` 在广告点击预测中首创 `GBDT+LR` 方案（[论文《**Practical Lessons from Predicting Clicks on Ads at Facebook**》](https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/  )），利用 `GBDT` 生成组合特征提升 `LR` 效果，通过捕捉用户行为和广告特征的复杂关系，显著提高了点击率预测的准确性。

### 6.2 金融风控

在信用评分、欺诈检测等场景下，`GBDT` 特征提取可用于增强模型解释性，确保业务合规，帮助金融机构理解模型决策依据，满足监管要求。

### 6.3 结合深度学习

微软提出的 `DeepGBM` 框架，将 `GBDT` 与神经网络联合训练，实现特征交互的端到端学习，充分发挥两者优势，提升模型性能。

## 7. 总结与展望

`GBDT` 作为一种强大的机器学习算法，不仅能用于分类和回归，还能高效地进行特征提取。通过叶子索引编码、特征重要性分析和 `SHAP` 解释，`GBDT` 在特征工程中展现了极大的优势。

注意事项：

* 叶子编码可能导致维度爆炸，需配合降维或正则化，如采用主成分分析（PCA）或 L1 正则化减少特征维度  
* 不同库的特征重要性计算逻辑不同，需参考文档，理解其统计方式和适用场景  

未来方向：

* `GBDT` 与图神经网络的结合，探索图结构数据中的复杂关系  
* 自动化特征工程框架（如 `AutoFeat`），进一步提升特征提取的效率和智能化水平


## 附录 - 叶子索引编码（Leaf Index Encoding）示例

假设我们有一个训练好的 `GBDT` 模型，由 **3 棵决策树** 组成，每棵树的叶子节点编号如下：

- **第一棵树**：`[0, 1, 2, 3]`
- **第二棵树**：`[0, 1]`
- **第三棵树**：`[0, 1, 2]`

### 步骤 1：确定样本的叶子索引

假设一个样本 `x`，在 `GBDT` 预测后落入以下叶子节点：

- **第一棵树** → 叶子节点 `2`
- **第二棵树** → 叶子节点 `1`
- **第三棵树** → 叶子节点 `0`

该样本的叶子索引表示为：

```python
[2, 1, 0]
```

### 步骤 2：One-Hot 编码

由于叶子索引是离散变量，可以使用 `One-Hot` 编码：

* 第一棵树（4 个叶子节点）：2 编码为 `[0, 0, 1, 0]`
* 第二棵树（2 个叶子节点）：1 编码为 `[0, 1]`
* 第三棵树（3 个叶子节点）：0 编码为 `[1, 0, 0]`

最终，该样本的 One-Hot 编码特征 为：

```python
[0, 0, 1, 0,  # 第一棵树
 0, 1,        # 第二棵树
 1, 0, 0]     # 第三棵树
```

### 步骤 3：使用嵌入（Embedding）

对于大规模数据集，`One-Hot` 编码可能会导致特征维度过高，因此可以用嵌入层（`Embedding`） 将叶子索引映射到低维稠密向量。例如，设 `Embedding` 维度为 4，则：

* `2 → [0.1, -0.3, 0.5, 0.7]`
* `1 → [0.4, -0.2, 0.6, 0.8]`
* `0 → [-0.1, 0.2, 0.3, -0.4]`

最终，该样本的**嵌入向量**为：

```python
[0.1, -0.3, 0.5, 0.7,  # 第一棵树
 0.4, -0.2, 0.6, 0.8,  # 第二棵树
 -0.1, 0.2, 0.3, -0.4] # 第三棵树
```

### 总结

叶子索引编码通过 `One-Hot` 或嵌入方式，将 `GBDT` 结构信息转换成可用于深度学习的特征表示。这种方法能够捕捉样本在树模型中的非线性特征，从而提高模型的预测能力。

