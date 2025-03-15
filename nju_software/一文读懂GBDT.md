# 一文读懂GBDT

## 一、定义
`GBDT`（`Gradient Boosting Decision Tree`）即**梯度提升决策树**，是一种基于 `Boosting`集成学习思想的加法模型，训练时采用前向分布算法进行贪婪的学习，每次迭代都学习一棵`CART`树来拟合之前`t-1`棵树的预测结果与训练样本真实值的**残差**。它属于 `Boosting` 方法的一种，通过构建多个决策树来逐步修正之前模型的错误，从而提升模型整体的预测性能。

## 二、数学原理
`GBDT`的数学原理主要包含以下几个部分：

1. **初始化模型**：通常使用公式$$f_0(x) = \arg\min_{c} \sum_{i=1}^{N} L(y_i, c)$$来初始化模型，其中$f_0(x)$表示初始模型的预测值，$c$是一个常数，$L(y_i, c)$表示损失函数。
2. 迭代过程：
   - **拟合残差**：以当前残差$r_{t-1}$为学习目标，训练一个弱学习器（决策树）$h_t(x)$，使其尽可能拟合$r_{t-1}$。
   - **计算步长（学习率）**：确定一个正的常数$\alpha_t$，通常通过交叉验证或线性搜索找到最佳值。
   - **更新预测**：将新学习到的决策树加入到累加函数中，更新预测值为$F_t(x)=F_{t-1}(x)+\alpha_th_t(x)$。
   - **计算新残差**：根据新的预测值计算残差$r_t=y-F_t(x)$。
3. **停止条件**：当达到预定的迭代次数T或残差变化小于阈值时停止迭代，最终的预测模型为$F(x)=\sum_{t=1}^{T}\alpha_th_t(x)$。
4. **负梯度**：在每一轮迭代中，计算损失函数的负梯度来指导下一棵树的生长。对于回归任务，当损失函数选用均方误差损失时，负梯度就是真实值与预测值的残差。

## 三、示例推导
假设我们有一个简单的回归问题，数据集如下：

| 特征1 (x1) | 目标值 (y) |
|------------|------------|
| 1          | 14         |
| 2          | 16         |
| 3          | 24         |
| 4          | 26         |

我们将使用`GBDT`来解决这个问题，设置迭代次数 $ T = 2 $。

### 第一步：初始化模型

我们使用均方误差损失函数 $ L(y, F) = (y - F)^2 $。初始模型 $ F_0(x) $ 为所有目标值的平均值：

$$ F_0(x) = \frac{14 + 16 + 24 + 26}{4} = 20 $$

### 第二步：迭代过程

#### 迭代1
1. **计算残差**:
   
   $$ r_{1i} = y_i - F_0(x_i) $$
   $$ r_{11} = 14 - 20 = -6 $$
   $$ r_{12} = 16 - 20 = -4 $$
   $$ r_{13} = 24 - 20 = 4 $$
   $$ r_{14} = 26 - 20 = 6 $$

2. **训练决策树** $ h_1(x) $ 来拟合残差 $ r_{1i} $。假设我们得到的树如下：

   - 如果 $ x_1 \leq 2.5 $，则 $ h_1(x) = -5 $
   - 如果 $ x_1 > 2.5 $，则 $ h_1(x) = 5 $

3. **计算步长** $ \alpha_1 $。我们使用线性搜索来找到最小化损失函数的 $ \alpha_1 $：

   $$
   F_1(x) = F_0(x) + \alpha_1 h_1(x)
   $$
   
   $$
   \text{损失} = \sum_{i=1}^{4} (y_i - F_1(x_i))^2
   $$

   通过尝试不同的 $ \alpha_1 $ 值，我们发现 $ \alpha_1 = 1 $ 时损失最小。

4. **更新预测值**：
   
   $$ F_1(x) = 20 + 1 \cdot h_1(x) $$
   $$ F_1(x_1) = 20 - 5 = 15 $$
   $$ F_1(x_2) = 20 - 5 = 15 $$
   $$ F_1(x_3) = 20 + 5 = 25 $$
   $$ F_1(x_4) = 20 + 5 = 25 $$

5. **计算新残差**：
   
   $$ r_{2i} = y_i - F_1(x_i) $$
   $$ r_{21} = 14 - 15 = -1 $$
   $$ r_{22} = 16 - 15 = 1 $$
   $$ r_{23} = 24 - 25 = -1 $$
   $$ r_{24} = 26 - 25 = 1 $$

#### 迭代2
1. **训练决策树** $ h_2(x) $ 来拟合残差 $ r_{2i} $。假设我们得到的树如下：

   - 如果 $ x_1 \leq 2.5 $，则 $ h_2(x) = -1 $
   - 如果 $ x_1 > 2.5 $，则 $ h_2(x) = 1 $

2. **计算步长** $ \alpha_2 $。我们使用线性搜索来找到最小化损失函数的 $ \alpha_2 $：

   $$
   F_2(x) = F_1(x) + \alpha_2 h_2(x)
   $$
   
   $$
   \text{损失} = \sum_{i=1}^{4} (y_i - F_2(x_i))^2
   $$

   通过尝试不同的 $ \alpha_2 $ 值，我们发现 $ \alpha_2 = 1 $ 时损失最小。

3. **更新预测值**：
   
   $$ F_2(x) = F_1(x) + 1 \cdot h_2(x) $$
   $$ F_2(x_1) = 15 - 1 = 14 $$
   $$ F_2(x_2) = 15 + 1 = 16 $$
   $$ F_2(x_3) = 25 - 1 = 24 $$
   $$ F_2(x_4) = 25 + 1 = 26 $$

### 第三步：停止条件
我们已经完成了 $ T = 2 $ 次迭代，所以停止迭代。最终的预测模型为：

$$
F(x) = F_2(x) = 20 + h_1(x) + h_2(x)
$$

### 最终预测模型

根据上面的推导，最终的预测模型可以表示为：

- 如果 $ x_1 \leq 2.5 $，则 $ F(x) = 20 - 5 - 1 = 14 $
- 如果 $ x_1 > 2.5 $，则 $ F(x) = 20 + 5 + 1 = 26 $

## 四、Python示例

以下是一个使用`sklearn`库实现`GBDT`的简单示例：

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from matplotlib import rcParams

# 配置 matplotlib 使用字体
rcParams['font.sans-serif'] = ['Heiti TC']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载Iris数据集
iris = load_iris()
X, y = iris.data, iris.target

# 数据集的基本信息
print("数据集形状:", X.shape)
print("特征名称:", iris.feature_names)
print("目标名称:", iris.target_names)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建GBDT分类器
gbdt = GradientBoostingClassifier(
    n_estimators=100,  # 树的数量
    learning_rate=0.1,  # 学习率
    max_depth=3,  # 树的最大深度
    random_state=42
)

# 训练模型
gbdt.fit(X_train, y_train)

# 预测
y_pred = gbdt.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}")

# 输出分类报告
print("分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 输出特征重要性
print("特征重要性:", gbdt.feature_importances_)
```

输出：

```text
数据集形状: (150, 4)
特征名称: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
目标名称: ['setosa' 'versicolor' 'virginica']
模型准确率: 1.0000
分类报告:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

特征重要性: [0.00135739 0.01465991 0.66567721 0.31830549]
```

## 五、总结
`GBDT`通过迭代地拟合残差来逐步提升模型的预测性能。每次迭代都学习一棵新的决策树来修正之前模型的错误，最终得到一个强大的预测模型。