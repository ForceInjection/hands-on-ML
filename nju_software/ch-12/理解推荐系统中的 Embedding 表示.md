# 理解推荐系统中的 Embedding 表示

在推荐系统中，**Embedding（嵌入表示）**是一种核心技术，用于将离散的、高维的类别信息（如用户 ID、商品 ID、性别、品类等）转化为**稠密的、低维的向量表示**，以便输入到模型中学习用户与物品之间的潜在关系。

---

## 1. 为什么推荐系统中需要 Embedding？

推荐系统中常见的原始特征大多是**离散型**，比如：

* 用户 ID（可能有上亿个）
* 商品 ID（成千上万个）
* 用户年龄段、性别、城市
* 商品类目、品牌、价格区间

这些特征通常是类别特征（`categorical features`），如果使用 `One-hot` 编码会导致：

* 维度爆炸（百万级稀疏向量）
* 无法刻画类别之间的相似性

为了解决这些问题，引入 **Embedding 技术**。

---

## 2. 什么是 Embedding？

**Embedding 就是一个查找表**，它把每个类别映射成一个固定长度的稠密向量。

### 2.1 举个例子

假设有 1000 个商品（商品 ID 从 0 到 999），我们设置 `embedding_dim = 32`，则：

* 每个商品 ID 会被映射为一个 32 维的向量
* 这个映射关系是通过一个 `1000 × 32` 的矩阵实现的
* 在训练过程中，这个矩阵会不断被优化，以学习到商品之间的语义关系

```python
# PyTorch 例子
embedding = nn.Embedding(num_embeddings=1000, embedding_dim=32)
item_vec = embedding(torch.tensor([42]))  # 获取商品 ID 为 42 的向量表示
```

---

## 3. 推荐系统中的典型 Embedding 用法

### 3.1 用户与物品的 Embedding

| 对象    | 描述       | 示例                        |
| ----- | -------- | ------------------------- |
| 用户 ID | 表示用户长期兴趣 | `user_embedding(user_id)` |
| 商品 ID | 表示商品特征向量 | `item_embedding(item_id)` |

然后通过以下方法计算匹配程度：

```python
score = dot(user_emb, item_emb)  # 向量内积
```

或用于更复杂模型，如 DNN、FM、Transformer 等。

### 3.2 多种特征的 Embedding

推荐系统通常使用多个嵌入输入：

* 用户特征：ID、性别、年龄段、城市
* 商品特征：ID、类目、品牌、价格
* 上下文特征：时间段、位置、设备类型

每个类别字段都可用 embedding 层转成向量，再拼接作为输入：

```python
features = [user_id_emb, gender_emb, item_id_emb, item_cat_emb, ...]
x = torch.cat(features, dim=1)
```

---

## 4. Embedding 的价值

| 作用   | 描述                             |
| ---- | ------------------------------ |
| 压缩维度 | 将稀疏的 One-hot 编码压缩为稠密向量         |
| 表达语义 | 学习用户/商品之间的隐式语义关系（如相似性）         |
| 可拓展  | 可用于构建深度模型（如 Wide\&Deep、DeepFM） |
| 支持召回 | Embedding 方便做向量召回（近邻搜索）        |

---

## 5. Embedding 和矩阵分解的联系

其实，**Embedding 本质上就是一种矩阵分解**：

* 用户矩阵 $P$ 和商品矩阵 $Q$ 可以看作 embedding 表
* 二者乘积 $P \cdot Q^T$ 预测评分，与 FM/SVD 等模型相似

在现代深度推荐系统中，Embedding 是构建神经网络推荐模型的基石。

---

## 6. 总结

| 关键词       | 含义                                              |
| --------- | ----------------------------------------------- |
| Embedding | 将类别变量映射成低维稠密向量                                  |
| 用途        | 表示用户/商品特征、上下文特征                                 |
| 优点        | 降维、高效、捕捉相似性                                     |
| 典型应用      | FM、Wide\&Deep、DIN、YouTube DNN、Transformer 推荐模型等 |

> 参考：[万字长文深入浅出文本嵌入（Text-Embedding）技术](https://mp.weixin.qq.com/s/npwT3_kaS5RDtYslz1caFQ)

---
