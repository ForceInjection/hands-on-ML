# 综合案例：电影推荐系统实战

## 1. 案例背景

本案例旨在构建一个完整的电影推荐系统，涵盖从数据处理、模型构建到评估的完整流程。我们将使用经典的 **MovieLens** 数据集的子集（或类似的评分数据）进行实验。

**核心目标**：

1. **数据探索 (EDA)**：理解长尾分布、稀疏性等推荐数据特征。
2. **模型实现**:
   - **Baseline**: 全局平均分、用户/物品平均分。
   - **协同过滤**: UserCF 和 ItemCF。
   - **矩阵分解**: SVD (使用 Surprise 库)。
3. **评估**: 使用 RMSE (Root Mean Square Error) 评估评分预测的准确性。

---

## 2. 数据集说明

案例包含以下数据文件：

- `recommendation-ratings-train.txt`: 训练集，包含用户对电影的评分。
- `recommendation-ratings-test.txt`: 测试集，用于评估模型性能。

**数据格式**:

```csv
userId,movieId,rating,timestamp
1,1,4,964982703
1,3,4,964981247
...
```

- `userId`: 用户 ID
- `movieId`: 电影 ID
- `rating`: 评分 (0.5 - 5.0)
- `timestamp`: 时间戳

---

## 3. 核心流程详解

### 3.1 数据加载与清洗

- 使用 `pandas` 读取 CSV 文件。
- 检查缺失值。
- 统计用户数、电影数、稀疏度 (Sparsity)。
  $$ \text{Sparsity} = 1 - \frac{\text{Ratings Count}}{\text{Users} \times \text{Items}} $$

### 3.2 探索性数据分析

- **长尾效应 (Long Tail)**: 绘制电影流行度曲线，观察是否符合“二八定律”（20% 的热门电影占据 80% 的交互）。
- **评分分布**: 用户的评分倾向（宽容 vs 苛刻）。

### 3.3 模型构建与对比

我们在 Notebook 中实现了以下模型：

#### A. 统计学基线 (Statistical Baselines)

- **Global Mean**: 预测所有评分为训练集的平均分。
- **User Mean**: 预测评分为该用户的平均分。
- **Item Mean**: 预测评分为该电影的平均分。

#### B. 协同过滤 (Collaborative Filtering)

- **User-Based CF**: 寻找相似用户。
- **Item-Based CF**: 寻找相似物品（通常效果优于 UserCF）。

#### C. 矩阵分解 (Matrix Factorization)

- 使用 `Surprise` 库的 `SVD` 算法。
- 超参数调优 (Grid Search): `n_factors` (隐因子数), `lr_all` (学习率), `reg_all` (正则化)。

### 3.4 评估结果示例

| 模型        | RMSE (越低越好) | 说明                       |
| :---------- | :-------------- | :------------------------- |
| Global Mean | 1.05            | 最差，仅作为 Baseline      |
| User Mean   | 0.98            | 考虑了用户偏差             |
| Item Mean   | 0.96            | 考虑了物品偏差             |
| **SVD**     | **0.87**        | 效果最佳，捕捉了隐语义特征 |

---

## 4. 代码运行指南

请打开 `综合案例_电影推荐.ipynb` Jupyter Notebook 进行交互式学习。

**依赖库**:

```bash
pip install numpy pandas matplotlib seaborn scikit-surprise
```

**关键代码片段**:

```python
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

# 1. 加载数据
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# 2. 划分数据集
trainset, testset = train_test_split(data, test_size=0.25)

# 3. 训练 SVD 模型
algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
algo.fit(trainset)

# 4. 预测与评估
predictions = algo.test(testset)
accuracy.rmse(predictions)
```

---

## 5. 扩展思考

1. **冷启动**: 如何处理测试集中出现的新用户或新电影？(SVD 会使用全局平均值填充)
2. **隐式反馈**: 如果只有点击数据（0/1），应该使用什么模型？(ALS / SVD++)
3. **模型融合**: 能否将 SVD 和 ItemCF 的结果加权融合以提升 RMSE？
