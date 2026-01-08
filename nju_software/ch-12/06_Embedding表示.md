# Embedding 表示 (Embedding Representation)

在现代深度学习推荐系统 (Deep Learning Recommendation System, DLRS) 中，**Embedding (嵌入)** 是连接稀疏特征与稠密神经网络的桥梁，被誉为深度学习推荐系统的“基石”。

原始的推荐数据（如用户 ID、物品 ID、类别标签）通常是**高维稀疏 (High-dimensional Sparse)** 的类别特征 (Categorical Features)。若直接使用 One-Hot 编码，维度可能达到数亿（例如用户 ID），导致“维度灾难”和计算不可行。**Embedding** 技术将这些离散的符号映射到**低维稠密 (Low-dimensional Dense)** 的实数向量空间。在这个空间中，语义相似的实体（如看过同样电影的用户，或风格相似的电影）距离更近。

本章将深入剖析 Embedding 的核心原理，从数学上的查表操作到 Item2Vec 算法，并结合 PyTorch 代码展示如何在工程中实现高效的 Embedding 层及变长序列处理。

---

## 1. 核心原理

Embedding 并非黑科技，其本质是**降维**。从数学视角看，它是一个全连接层；从工程视角看，它是一次高效的查表操作。本节将拆解这一过程，并介绍如何利用 Word2Vec 思想生成物品向量。

### 1.1 查表操作

从数学上讲，Embedding 层本质上是一个全连接层 (Fully Connected Layer)，但由于输入是 One-Hot 向量，矩阵乘法退化为**查表 (Lookup)** 操作。

假设词表大小为 $N$，Embedding 维度为 $K$。

- **权重矩阵**: $W \in \mathbb{R}^{N \times K}$
- **输入**: ID $i$ (对应 One-Hot 向量 $x_i$)
- **输出**: $e_i = x_i \cdot W = W[i, :]$ (即取出第 $i$ 行)

这种设计使得前向传播和反向传播非常高效，只更新涉及的行，而无需进行全矩阵乘法。

### 1.2 示例：从 ID 到向量

为了更直观地理解，让我们看一个具体的数值例子。

假设我们有 1000 个商品（ID 范围 0~999），设定 Embedding 维度为 4。
此时，Embedding 矩阵 $W$ 的形状为 $1000 \times 4$。

如果我们要获取 **ID 为 42** 的商品的向量表示：

1. **输入**: 索引 `42`。
2. **操作**: 直接从矩阵 $W$ 中取出第 42 行。
3. **结果**: 得到一个 4 维的稠密向量，例如 `[0.12, -0.45, 0.88, 0.03]`。

这个向量就代表了商品 42 的所有特征信息。在训练过程中，这 4 个数值会随着梯度下降不断更新，最终学会表达该商品的语义（如“电子产品”、“高价”等）。

### 1.3 Item2Vec: 物品向量化

受 NLP 领域 **Word2Vec** (Skip-gram) 的启发，我们可以利用用户行为序列来学习物品的 Embedding，称为 **Item2Vec**。

- **假设**: 经常在同一个 Session 中出现的物品具有相似性（类似上下文中的单词）。
- **做法**: 将用户浏览序列看作“句子”，物品看作“单词”，训练 Word2Vec 模型。
- **产出**: 物品的静态向量，可用于计算物品相似度 (Item-to-Item Retrieval)。

目标函数 (Negative Sampling):

$$
J = \log \sigma(u_i \cdot v_j) + \sum_{k=1}^{K} \mathbb{E}_{v_k \sim P_n(v)} [\log \sigma(-u_i \cdot v_k)]
$$

其中：

- $u_i$ 是中心词（物品）向量，$v_j$ 是上下文物品向量。
- $\sigma(x)$ 是 Sigmoid 函数，用于将内积映射为概率。
- $P_n(v)$ 是负采样分布（通常根据物品热度的 $0.75$ 次幂采样）。

---

## 2. 工程实现

理论的落地离不开代码。在实际工程中，我们不仅要处理单一 ID 的 Embedding，还要处理用户变长的行为序列（如历史点击记录）。本节将使用 PyTorch 构建一个完整的 Embedding 模块，包含初始化、查表以及序列 Pooling 策略。

### 2.1 PyTorch 实现 Embedding 层

以下代码展示了如何在 PyTorch 中定义 Embedding 层，并处理多值特征（如用户的历史行为序列）。

```python
import torch
import torch.nn as nn

class RecommendationModel(nn.Module):
    """
    简单的推荐模型示例，展示 Embedding 层的使用
    """
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(RecommendationModel, self).__init__()

        # 1. 定义 Embedding 表
        # num_embeddings: 词表大小 (用户数或物品数)
        # embedding_dim: 嵌入向量维度
        self.user_embedding = nn.Embedding(num_embeddings=num_users,
                                           embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items,
                                           embedding_dim=embedding_dim)

        # 初始化权重
        # 通常使用 Xavier (Glorot) 初始化或正态分布初始化
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user_ids, item_ids):
        """
        前向传播

        Parameters
        ----------
        user_ids : tensor, shape [batch_size]
        item_ids : tensor, shape [batch_size]
        """
        # 2. 查表 (Lookup)
        # user_embeds: [batch_size, embedding_dim]
        user_embeds = self.user_embedding(user_ids)
        # item_embeds: [batch_size, embedding_dim]
        item_embeds = self.item_embedding(item_ids)

        # 3. 计算交互 (例如点积)
        # logits: [batch_size]
        # 逐元素相乘后在 dim=1 求和，即内积
        logits = (user_embeds * item_embeds).sum(dim=1)
        return logits

# 示例使用
model = RecommendationModel(num_users=10000, num_items=5000, embedding_dim=64)
user_input = torch.tensor([1, 2, 3])
item_input = torch.tensor([10, 20, 30])
output = model(user_input, item_input)
print(f"Prediction Scores: {output.detach().numpy()}")
```

### 2.2 处理变长序列

对于用户的历史行为序列（如最近看过的 5 个商品），长度不固定。通常采用 **Pooling** 策略将其压缩为固定长度向量。

常见的 Pooling 方法：

1. **Sum Pooling**: 所有向量相加。
2. **Average Pooling**: 所有向量求平均 (最常用，能消除序列长度影响)。
3. **Attention Pooling**: 加权求和 (DIN 模型的核心)。

```python
class SequencePoolingLayer(nn.Module):
    def __init__(self, mode='mean'):
        super(SequencePoolingLayer, self).__init__()
        self.mode = mode

    def forward(self, sequence_embeds, mask):
        """
        Parameters
        ----------
        sequence_embeds: [batch_size, max_len, dim]
            序列中每个物品的 Embedding
        mask: [batch_size, max_len]
            掩码矩阵 (1 for valid, 0 for padding)
        """
        if self.mode == 'mean':
            # 1. 对应位置相乘，去除 Padding 的影响
            # mask.unsqueeze(-1) shape: [batch_size, max_len, 1]
            valid_embeds = sequence_embeds * mask.unsqueeze(-1)

            # 2. 求和
            sum_embeds = torch.sum(valid_embeds, dim=1)

            # 3. 计算真实长度 (避免除以 0，加一个小常数)
            lengths = torch.sum(mask, dim=1).unsqueeze(-1).float()
            return sum_embeds / (lengths + 1e-8)

        elif self.mode == 'sum':
            return torch.sum(sequence_embeds * mask.unsqueeze(-1), dim=1)
```

> **提示**: PyTorch 提供了 `nn.EmbeddingBag`，它将 `Embedding` 和 `Sum/Mean Pooling` 合并为一个操作，性能比手动实现快得多，非常适合处理变长序列特征。

### 2.3 多域特征融合

现代推荐系统（如 DeepFM, Wide&Deep）通常包含多种特征（用户 ID、用户城市、物品 ID、物品类目等）。标准做法是将所有特征的 Embedding **拼接 (Concatenate)** 后输入 MLP。

```python
class MultiFieldModel(nn.Module):
    def __init__(self, field_dims, embed_dim=16):
        super().__init__()
        # ModuleList 存储每个域的 Embedding 层
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=f_dim, embedding_dim=embed_dim)
            for f_dim in field_dims
        ])
        # MLP 部分
        self.mlp = nn.Sequential(
            nn.Linear(len(field_dims) * embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, num_fields]
        embeds = []
        for i, layer in enumerate(self.embeddings):
            embeds.append(layer(x[:, i]))
        
        # 拼接: [batch_size, num_fields * embed_dim]
        concat_embed = torch.cat(embeds, dim=1)
        return torch.sigmoid(self.mlp(concat_embed))
```

### 2.4 嵌入初始化

Embedding 的初始化对模型收敛速度和最终效果有很大影响。

- **随机初始化**:
  - `xavier_uniform` / `xavier_normal`: 保持输入输出方差一致，最常用。
  - `normal(0, 0.01)`: 简单的正态分布。
- **预训练初始化 (Pre-trained)**:
  - 利用 **Item2Vec** 或 **Graph Embedding** (如 DeepWalk) 预先训练好物品向量。
  - 将预训练向量赋值给 Embedding 层，并选择是否冻结 (`freeze=True`) 或微调 (`fine-tune`)。

---

## 3. 进阶技巧与挑战

随着数据规模的爆炸式增长，简单的 Embedding 查表在海量 ID 面前显得力不从心。内存瓶颈、新 ID 的冷启动以及在线检索的实时性，都是必须解决的工程难题。

### 3.1 维度灾难与特征哈希

当 ID 数量极大（如 10 亿用户）时，单机内存无法存下 Embedding 表。
**Feature Hashing (Hashing Trick)** 是常用解决方案：

- 不直接映射 ID，而是将 ID 经过 Hash 函数映射到较小的空间 (Bucket Size)。
- 例如：`idx = hash(user_id) % 1000000`。
- **代价**: Hash 冲突会带来少量信息损失，但通常可接受。

在工程实践中，通常将 Hashing 与 `nn.EmbeddingBag` 结合使用，以极低的内存占用处理海量稀疏特征。

### 3.2 冷启动与 OOV (Out-Of-Vocabulary) 问题

对于新出现的 ID（未在训练集中出现）：

1. **默认向量**: 映射到一个特定的 `<UNK>` (Unknown) 向量（通常是零向量或随机向量）。
2. **Side Information**: 利用图像、文本等内容特征生成 Embedding，而不是仅依赖 ID。

### 3.3 向量检索

训练好 Embedding 后，在线服务时如何快速找到与用户向量最相似的 Top-K 物品？
暴力计算复杂度为 $O(N)$，不可接受。

工程上使用 **ANN (Approximate Nearest Neighbor)** 算法：

- **Faiss (Facebook)**: 基于聚类或乘积量化 (PQ)，支持 GPU 加速。
- **HNSW (Hierarchical Navigable Small World)**: 基于图的索引，召回率高，性能极佳。
- **向量数据库**: 如 Milvus, Pinecone, Weaviate。

---

## 4. 总结

Embedding 是推荐系统的基石，它完成了从**符号空间**到**向量空间**的转换。

- **输入**: 稀疏 ID (User ID, Item ID, Category ID)。
- **过程**: 查表 (Lookup) -> 序列 Pooling -> 交互层。
- **输出**: 稠密向量，用于 MLP 输入或相似度检索。

掌握 Embedding 的初始化、训练、Pooling 以及线上检索 (ANN)，是构建深度推荐系统的必备技能。

---

## 5. 参考文献

1. **Mikolov, T., et al.** (2013). Distributed Representations of Words and Phrases and their Compositionality. _NIPS_.
2. **Covington, P., Adams, J., & Sargin, E.** (2016). Deep Neural Networks for YouTube Recommendations. _RecSys_.
3. **Facebook AI Research**. Faiss: A library for efficient similarity search.
