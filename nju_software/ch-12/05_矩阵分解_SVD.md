# 矩阵分解 (Matrix Factorization): SVD 与隐语义模型

在 [01\_推荐系统综述.md](01_推荐系统综述.md) 中我们提到，推荐系统的核心在于高效地连接用户与物品。**矩阵分解 (Matrix Factorization, MF)** 技术作为协同过滤的进阶形态，曾长期统治推荐系统竞赛（如 Netflix Prize），是推荐系统发展史上的里程碑。

早期的协同过滤主要基于**邻域** (Neighborhood-based)，如 UserCF 和 ItemCF。然而，随着数据规模的增长，基于邻域的方法面临着**稀疏性**和**扩展性**的挑战。矩阵分解应运而生，它通过将高维稀疏的评分矩阵分解为两个低维稠密矩阵的乘积，从而挖掘用户和物品的**隐语义特征 (Latent Factors)**。

本章将深入探讨矩阵分解的核心原理，从手工计算的直观理解出发，介绍传统的 **SVD (Singular Value Decomposition)**，演进到工业界广泛使用的 **FunkSVD**、**BiasSVD** 以及 **SVD++**，并结合代码示例展示如何在工程中实现这些算法。

---

## 2. 示例计算：手工算一次矩阵分解

为了更直观地理解矩阵分解如何通过“隐向量”来预测评分，而不被复杂的数学公式劝退，我们首先通过一个极简案例来手工演算一遍推荐流程。

### 2.1 场景设定

假设我们有如下用户-电影评分矩阵（部分缺失），我们的目标是预测这些“?”处的评分。

|     | 电影 M1 (动作) | 电影 M2 (爱情) | 电影 M3 (动作) | 电影 M4 (爱情) |
| :-- | :------------: | :------------: | :------------: | :------------: |
| U1  |       5        |       ?        |       3        |       ?        |
| U2  |       4        |       ?        |       ?        |       2        |
| U3  |       ?        |       4        |       ?        |       5        |
| U4  |       3        |       ?        |       4        |       ?        |

### 2.2 核心思想：用户兴趣 × 电影特征

矩阵分解的核心思想是：我们可以将用户对物品的评分看作是**两个向量的乘积**。

$$
\text{预测评分} = \text{用户兴趣向量} \cdot \text{电影特征向量}
$$

假设我们设定**隐特征 (Latent Factor)** 的维度 $k=2$，这两个维度可能（隐式地）代表“动作片指数”和“爱情片指数”。

**示例向量：**

- **用户 U1** (喜欢动作片): `[0.9, 0.1]`
- **电影 M1** (典型动作片): `[1.0, 0.1]`

**预测计算：**

$$
\hat{r}_{U1, M1} = 0.9 \times 1.0 + 0.1 \times 0.1 = 0.91
$$

这个分数很高，符合 U1 给 M1 打 5 分的实际情况。

### 2.3 潜在因子空间

如果我们把所有用户和电影都映射到这个 2 维空间：

- **U1 (0.9, 0.1)** 和 **M1 (1.0, 0.1)** 距离很近（向量夹角小），点积大。
- **U3 (0.1, 0.9)** (喜欢爱情片) 和 **M1 (1.0, 0.1)** 距离较远，点积小。

通过这种方式，我们将原本稀疏的评分矩阵，转换为了稠密的**用户矩阵 P** 和 **物品矩阵 Q**。

---

## 3. 理论基础

矩阵分解并非凭空产生，而是建立在深厚的线性代数基础之上。本节将回顾传统的奇异值分解，分析其在推荐场景下的局限性，并引出更适合处理稀疏数据的隐语义模型。

### 3.1 传统的奇异值分解

从线性代数的角度，任意一个 $m \times n$ 的实矩阵 $R$ 都可以分解为三个矩阵的乘积：

$$
R = U \Sigma V^T
$$

其中：

- $U$: $m \times m$ 的正交矩阵，列向量称为左奇异向量。
- $\Sigma$: $m \times n$ 的对角矩阵，对角线元素为奇异值，按降序排列。
- $V^T$: $n \times n$ 的正交矩阵，列向量称为右奇异向量。

在推荐系统中，如果我们取前 $k$ 个最大的奇异值，可以将 $R$ 近似为：

$$
R \approx U_k \Sigma_k V_k^T
$$

这本质上是一种**降维 (Dimensionality Reduction)** 技术。

**工程局限性**：

1. **缺失值问题**：传统 SVD 要求矩阵是稠密的，而推荐系统的评分矩阵极度稀疏 (99% 以上为空)。简单的填充 (Imputation) 会引入巨大的噪声和计算开销。
2. **计算复杂度**：SVD 的计算复杂度为 $O(mn^2)$ 或 $O(m^2n)$，对于百万级用户和物品的场景无法接受。

### 3.2 隐语义模型

为了解决上述问题，推荐系统中的“SVD”通常指代**基于梯度的矩阵分解**，即 **FunkSVD**。

其核心思想是：直接建模用户矩阵 $P$ ($m \times k$) 和物品矩阵 $Q$ ($n \times k$)，使得它们的乘积近似于原始评分矩阵 $R$ 中**已知**的部分。

$$
\hat{r}_{ui} = p_u \cdot q_i = \sum_{f=1}^{k} p_{u,f} q_{i,f}
$$

其中：

- $p_u$: 用户 $u$ 的 $k$ 维隐向量 (Latent Vector)。
- $q_i$: 物品 $i$ 的 $k$ 维隐向量。
- $k$: 隐特征的维度 (Hyperparameter)，通常取 10~100。

---

## 4. 核心算法演进

为了适应工业界大规模、高稀疏的数据环境，矩阵分解算法经历了多次迭代与进化。从最基础的 FunkSVD 到引入偏置项的 BiasSVD，再到融合隐式反馈的 SVD++，本节将梳理这一演进路径，解析每一步改进背后的动机与原理。

### 4.1 FunkSVD: 基础矩阵分解

Simon Funk 在 Netflix Prize 竞赛中提出了该方法。

**目标函数 (Objective Function)**:

$$
J = \sum_{(u,i) \in \mathcal{K}} (r_{ui} - p_u \cdot q_i)^2 + \lambda (\|p_u\|^2 + \|q_i\|^2)
$$

- $\mathcal{K}$: 已知评分的 (用户, 物品) 集合。
- $\lambda$: 正则化系数 (Regularization Term)，防止过拟合。

**优化方法 (Optimization)**:

通常使用 **随机梯度下降 (Stochastic Gradient Descent, SGD)**。
对于每个样本 $(u, i)$，误差 $e_{ui} = r_{ui} - p_u \cdot q_i$。

更新规则：

$$
p_u \leftarrow p_u + \eta (e_{ui} q_i - \lambda p_u) \\
q_i \leftarrow q_i + \eta (e_{ui} p_u - \lambda q_i)
$$

其中 $\eta$ 为学习率。

### 4.2 BiasSVD: 引入偏置项

实际场景中，不同用户和物品存在固有的偏差 (Bias)。

- 有些用户倾向于打高分（宽容型）。
- 有些物品本身质量很高，普遍分高。

**BiasSVD** 模型在 FunkSVD 基础上增加了偏置项：

$$
\hat{r}_{ui} = \mu + b_u + b_i + p_u \cdot q_i
$$

其中：

- $\mu$: 全局平均分。
- $b_u$: 用户 $u$ 的偏置。
- $b_i$: 物品 $i$ 的偏置。

**目标函数**:

$$
J = \sum_{(u,i) \in \mathcal{K}} (r_{ui} - (\mu + b_u + b_i + p_u \cdot q_i))^2 + \lambda (b_u^2 + b_i^2 + \|p_u\|^2 + \|q_i\|^2)
$$

这是目前工业界最通用的矩阵分解基线模型。

### 4.3 SVD++: 融合隐式反馈

BiasSVD 仅利用了显式评分 (Explicit Feedback)。实际上，用户的浏览、点击等**隐式反馈 (Implicit Feedback)** 也能反映兴趣。

SVD++ 引入了 $y_j$ 向量，表示用户有过交互的物品 $j$ 对用户兴趣的贡献：

$$
\hat{r}_{ui} = \mu + b_u + b_i + q_i^T (p_u + |N(u)|^{-\frac{1}{2}} \sum_{j \in N(u)} y_j)
$$

- $N(u)$: 用户 $u$ 有过交互的物品集合。
- $|N(u)|^{-\frac{1}{2}}$: 归一化项。

---

## 5. 工程实践

理论落地不仅需要公式推导，更需要高效的代码实现。本节将分别通过原生 Python 代码（用于理解原理）和工业级库 Surprise（用于生产实践）来展示矩阵分解算法的构建过程，并探讨大规模场景下的并行化方案。

### 5.1 Python 实现 BiasSVD（原理演示）

为了将 4.2 节的数学公式转化为代码，我们使用 NumPy 手写一个 BiasSVD。

> **注意**：这段代码旨在**白盒演示**算法内部的梯度下降过程，帮助理解参数是如何更新的。在实际生产环境中，请直接使用 5.2 节介绍的 `Surprise` 库或 Spark ALS，它们经过了高度的性能优化（C++/Scala 实现）。

```python
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class BiasSVD(BaseEstimator, RegressorMixin):
    """
    BiasSVD 矩阵分解算法实现 (教学演示版)
    """
    def __init__(self, n_factors=20, n_epochs=20, lr=0.01, reg=0.02, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state

    def fit(self, X, y):
        np.random.seed(self.random_state)

        # 1. ID 映射 (Mapping): 将原始 ID 转为 0 ~ N-1 的矩阵索引
        self.user_map = {u: i for i, u in enumerate(np.unique(X[:, 0]))}
        self.item_map = {i: j for j, i in enumerate(np.unique(X[:, 1]))}

        n_users, n_items = len(self.user_map), len(self.item_map)

        # 2. 参数初始化 (Initialization)
        self.mu = np.mean(y)                   # 全局平均分
        self.bu = np.zeros(n_users)            # 用户偏置
        self.bi = np.zeros(n_items)            # 物品偏置
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors)) # 用户隐矩阵
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors)) # 物品隐矩阵

        # 3. SGD 训练 (Stochastic Gradient Descent)
        for epoch in range(self.n_epochs):
            total_loss = 0
            for k in range(len(y)):
                u_orig, i_orig = X[k]
                r = y[k]

                u, i = self.user_map[u_orig], self.item_map[i_orig]

                # [核心逻辑] 前向传播：计算预测值
                # Formula: \hat{r}_{ui} = \mu + b_u + b_i + p_u \cdot q_i
                dot = np.dot(self.P[u], self.Q[i])
                pred = self.mu + self.bu[u] + self.bi[i] + dot

                # 计算误差
                err = r - pred

                # [核心逻辑] 反向传播：更新参数
                # Formula: b_u <- b_u + \eta * (err - \lambda * b_u)
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                # 更新隐向量 P 和 Q
                pu_old = self.P[u].copy() # 暂存旧值
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * pu_old - self.reg * self.Q[i])

                total_loss += err**2

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs} - MSE: {total_loss/len(y):.4f}")

        return self


    def predict(self, X):
        """
        预测评分

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
            要预测的 [user_id, item_id] 对

        Returns
        -------
        preds : np.array
            预测评分数组
        """
        preds = []
        for u_orig, i_orig in X:
            # 处理冷启动：如果是新用户或新物品，返回全局平均分
            if u_orig not in self.user_map or i_orig not in self.item_map:
                preds.append(self.mu)
                continue

            u = self.user_map[u_orig]
            i = self.item_map[i_orig]

            pred = self.mu + self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])
            preds.append(pred)
        return np.array(preds)
```

### 5.2 使用 Surprise 库 (推荐)

在生产环境中，建议使用成熟的库如 [Surprise](http://surpriselib.com/)，它进行了大量 C++ 层面的优化。

```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# 1. 准备数据
# 假设 ratings_df 有 columns: ['userID', 'itemID', 'rating']
# Reader 指定评分范围
reader = Reader(rating_scale=(1, 5))
# 这里假设 ratings_df 已经加载了数据
# data = Dataset.load_from_df(ratings_df[['userID', 'itemID', 'rating']], reader)

# 为了演示，我们使用 Surprise 内置的 ml-100k 数据集（会自动下载）
data = Dataset.load_builtin('ml-100k')

# 2. 初始化模型 (BiasSVD)
# n_factors: 隐因子数目
# n_epochs: 迭代次数
# lr_all: 学习率
# reg_all: 正则化参数
algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)

# 3. 交叉验证
# cv=5 表示 5 折交叉验证
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# 4. 全量训练与预测
trainset = data.build_full_trainset()
algo.fit(trainset)

# 预测用户 '196' 对物品 '302' 的评分 (注意 ID 是字符串)
prediction = algo.predict(uid='196', iid='302')
print(f"预测评分: {prediction.est:.2f}")
```

### 5.3 扩展性讨论

对于海量数据（如 1 亿条以上评分），单机 SGD 训练会非常慢。

**ALS (Alternating Least Squares, 交替最小二乘法)** 是另一种优化方法。

- **原理**：固定 $P$，目标函数变为关于 $Q$ 的二次函数，有闭式解；然后固定 $Q$，求 $P$。交替进行。
- **优势**：可以并行化 (Parallelizable)。Spark MLlib 中的 `ALS` 是处理大规模矩阵分解的标准工具。

---

## 6. 总结与扩展阅读

矩阵分解作为推荐系统发展史上的里程碑技术，虽然面临深度学习的挑战，但其思想依然深刻影响着当今的 Embedding 技术。本节将对不同变种算法进行横向对比，并给出进一步学习的参考文献。

### 6.1 算法对比

| 算法        | 特点                             | 适用场景                       |
| :---------- | :------------------------------- | :----------------------------- |
| **SVD**     | 数学严谨，要求稠密矩阵           | 理论分析，不直接用于大规模推荐 |
| **FunkSVD** | 使用 SGD 求解，处理稀疏矩阵      | 基础推荐 Baseline              |
| **BiasSVD** | 增加 Bias 项，拟合能力更强       | **工业界最常用**               |
| **SVD++**   | 增加隐式反馈，精度最高，但训练慢 | 追求极致精度的场景             |

### 6.2 参考文献

1. **Koren, Y., Bell, R., & Volinsky, C.** (2009). Matrix Factorization Techniques for Recommender Systems. _IEEE Computer_.
2. **Netflix Prize Documentation**. The BellKor Solution to the Netflix Grand Prize.
3. **Surprise Library Documentation**: [http://surpriselib.com/](http://surpriselib.com/)
