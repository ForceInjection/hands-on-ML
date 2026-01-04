# Stacking 算法实战指南

本文档旨在作为《[一文深入了解机器学习之集成学习](./一文深入了解机器学习之集成学习.md)》的深度实战补充，专注于 Stacking 算法的工程实现与细节落地。我们将跳出简单的 API 调用，深入解析 `mlxtend` 库背后的数据流转逻辑，并通过两个经典案例展示如何构建无偏的元特征以及有效地组合异构模型。

文档结构包含四部分：

1. **工程机制解析**：剖析 Stacking 中最关键的 K 折交叉验证堆叠策略，解释它是如何防止数据泄露的。
2. **环境准备**：列出实战所需的依赖库。
3. **多分类实战（Iris）**：演示如何利用异构基学习器（KNN、RF、NB）构建互补的决策边界。
4. **二分类实战（Heart Disease）**：展示在真实业务场景中，如何结合数据预处理与 Stacking 提升预测性能。

通过本指南，读者将掌握 Stacking 的完整工作流，具备在实际项目中手写或调优 Stacking 模型的能力。

---

## 一、工程机制解析：交叉验证生成元特征

Stacking 的本质是**学习“如何利用基模型的预测值”**。但在训练元学习器（Meta-Learner）时，我们不能直接使用基模型在训练集上的预测结果，因为这些结果通常是过拟合的（基模型已经“见过”这些数据）。

为了解决这个问题，Stacking 采用了**K 折交叉验证（K-Fold CV）**策略来生成“干净”的元特征：

1. **划分数据**：将训练集分为 K 份（例如 K=5）。
2. **循环预测**：
   - 每次取 K-1 份作为训练子集，训练基模型。
   - 用训练好的基模型预测剩下的 1 份验证子集。
   - 重复 K 次，直到所有样本都作为验证集被预测过一次。
3. **拼接特征**：将这 K 次的预测结果拼接起来，形成一个与原始训练集等长的预测向量。这个向量就是该基学习器提供的**元特征**。
4. **元模型训练**：将所有基学习器的元特征组合成新的特征矩阵，作为输入来训练第二层的元学习器。

`mlxtend` 库的 `StackingClassifier` 内部自动封装了这一过程，确保了数据流转的正确性。

---

## 二、实战案例一：鸢尾花分类（多分类）

本案例演示如何结合 KNN、随机森林和朴素贝叶斯三个差异较大的基学习器，构建一个强大的 Stacking 分类器。

**对应 Notebook**: [Stacking_iris.ipynb](./Stacking_iris.ipynb)

### 2.1 核心代码解析

首先，定义基学习器（第一层）和元学习器（第二层）。

```python
# 导入模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier

# 定义基学习器
clf1 = KNeighborsClassifier(n_neighbors=3)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = GaussianNB()

# 定义元学习器（通常使用简单的线性模型）
lr = LogisticRegression()

# 构建 Stacking 分类器
# classifiers: 第一层基学习器列表
# meta_classifier: 第二层元学习器
# use_probas=False (默认): 使用基学习器的类别预测值作为元特征
# average_probas=False (默认): 不对概率值取平均
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                          meta_classifier=lr)
```

### 2.2 模型训练与评估

通过交叉验证对比单模型与集成模型的性能。这里需要注意，外层的 `cross_val_score` 是用来评估整体模型性能的，而 `StackingClassifier` 内部还有一层交叉验证是用来生成元特征的（默认 `use_features_in_secondary=False`）。

```python
# 训练与评估循环
for clf, label in zip([clf1, clf2, clf3, sclf],
                      ['KNN', 'Random Forest', 'Naive Bayes', 'Stacking Classifier']):

    # 计算交叉验证准确率
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))
```

### 2.3 决策边界可视化

通过 `mlxtend.plotting.plot_decision_regions` 可以直观地看到 Stacking 如何融合不同模型的决策边界，形成更复杂的分类面。

---

## 三、实战案例二：心脏病预测（二分类）

本案例展示了在真实医疗数据集中，如何进行数据预处理并应用 Stacking 提升预测准确率。

**对应 Notebook**: [Stacking_heart.ipynb](./Stacking_heart.ipynb)

### 3.1 数据预处理的重要性

对于 KNN 等距离敏感模型，特征标准化至关重要：

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
# 选取数值型特征进行标准化
var_transform = ['thalach', 'age', 'trestbps', 'oldpeak', 'chol']
X_train[var_transform] = sc.fit_transform(X_train[var_transform])
X_test[var_transform] = sc.transform(X_test[var_transform])
```

### 3.2 构建 Stacking 模型

在本例中，我们使用 KNN 和朴素贝叶斯作为基学习器，逻辑回归作为元学习器。

```python
# 定义模型
KNC = KNeighborsClassifier(n_neighbors=3)
NB = GaussianNB()
lr = LogisticRegression()

# 构建 Stacking 分类器
sclf = StackingClassifier(classifiers=[KNC, NB],
                          meta_classifier=lr)

# 训练并预测
sclf.fit(X_train, y_train)
pred_stack = sclf.predict(X_test)
print('Stacking Classifier Accuracy:', accuracy_score(y_test, pred_stack))
```

---

## 四、总结

Stacking 作为集成学习的“集大成者”，其核心魅力在于能够从不同模型的预测中挖掘出更深层的模式。通过本文的原理解析与实战演练，我们应当深刻理解以下几个关键点，以便在未来的项目中灵活运用：

1. **多样性（Diversity）**：Stacking 的效果依赖于基学习器的差异性。如案例一中混合了基于距离（KNN）、基于树（RF）和基于概率（NB）的模型。
2. **元学习器选择**：第二层模型通常选择简单的线性模型（如逻辑回归），以避免过拟合。
3. **交叉验证机制**：`StackingClassifier` 内部默认使用交叉验证来生成元特征（`use_probas=False` 时），有效防止数据泄露。这是手动实现 Stacking 时最容易出错的地方。
4. **计算代价**：Stacking 需要训练多个基模型，且内部涉及交叉验证，训练时间通常是单模型的 K 倍。因此在工程落地时需要权衡精度提升与计算成本。

通过这两个实战案例，我们可以看到 Stacking 是一种灵活且强大的集成策略，能够充分利用不同算法的优势来提升最终的预测性能。

---
