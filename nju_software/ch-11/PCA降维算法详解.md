# 降维算法详解

引言
-----------

在互联网大数据场景下，我们经常需要面对高维数据，在对这些数据做分析和可视化的时候，我们通常会面对「高维」这个障碍。在数据挖掘和建模的过程中，高维数据也同样带来大的计算量，占据更多的资源，而且许多变量之间可能存在相关性，从而增加了分析与建模的复杂性。

我们希望找到一种方法，在对数据完成降维「压缩」的同时，尽量减少信息损失。由于各变量之间存在一定的相关关系，因此可以考虑将关系紧密的变量变成尽可能少的新变量，使这些新变量是两两不相关的，那么就可以用较少的综合指标分别代表存在于各个变量中的各类信息。机器学习中的降维算法就是这样的一类算法。

**主成分分析（Principal Components Analysis，简称PCA）是最重要的数据降维方法之一**。在数据压缩消除冗余和数据噪音消除等领域都有广泛的应用。本篇我们来展开讲解一下这个算法。


1.PCA与最大可分性
-----------

对于 $X = \begin {bmatrix} x_1 \\ x_2 \\ ... \\ x_n \end{bmatrix}$，$X \in R^n$。我们希望 $X$ 从 $n$ 维降到 $n^{'}$ 维，同时希望信息损失最少。比如，从 $n = 2$ 维降到 $n^{'} = 1$。

![Image 9: PCA降维算法; PCA与最大可分性;](https://img-blog.csdnimg.cn/img_convert/e7e627e2b55e32308e51ee253072b7c4.png)

左图为一个典型的例子，假如我们要对一系列人的样本进行数据降维（每个样本包含「身高」「体重」两个维度）。右图我们既可以降维到第一主成分轴，也可以降维到第二主成分轴。

哪个主成分轴更优呢？从直观感觉上，我们会认为「第一主成分轴」优于「第二主成分轴」，因为它比较大程度保留了数据之间的区分性（保留大部分信息）。

对`PCA`算法而言，我们希望找到小于原数据维度的若干个投影坐标方向，把数据投影在这些方向，获得压缩的信息表示。下面我们就一步一步来推导一下 `PCA` 算法原理。

2.基变换
-----

先来复习一点点数学知识。我们知道要获得原始数据 $X$ 新的表示空间 $Y$，最简单的方法是对原始数据进行线性变换（也叫做基变换） $Y = PX$。其中，$X$ 是原始样本，$P$ 是基向量，$Y$ 是新表达。

数学表达为：

$\begin{bmatrix} p_1 \\ p_2 \\ \vdots \\ p_r \end{bmatrix}_{r \times n}  \begin{bmatrix} x_1 & x_2 & \cdots & x_m \end{bmatrix}_{n \times m} =  \begin{bmatrix} p_1 x_1 & p_1 x_2 & \cdots & p_1 x_m \\ p_2 x_1 & p_2 x_2 & \cdots & p_2 x_m \\ \vdots & \vdots & \ddots & \vdots \\ p_r x_1 & p_r x_2 & \cdots & p_r x_m\end{bmatrix}_{r\times m}$

*   其中 $p_i$ 是行向量，表示第 $i$ 个基；
    
*   $x_j$ 是一个列向量，表示第 $j$ 个原始数据记录。
    

当 $r < n$ 时，即「基的维度<数据维度」时，可达到降维的目的，即 $X \in R^{n \times m} \rightarrow Y \in R^{r \times m }$。

![Image 23: PCA降维算法; 线性变换 / 基变换;](https://img-blog.csdnimg.cn/img_convert/6779697175868b2c40027990d8db7e79.png)

以直角坐标系下的点 $(3,2)$ 为例，要把点 $(3,2)$ 变换为新基上的坐标，就是用 $(3,2)$ 与第一个基做内积运算，作为第一个新的坐标分量，然后用 $(3,2)$ 与第二个基做内积运算，作为第二个新坐标的分量。

![Image 28: PCA降维算法; 线性变换 / 基变换;](https://img-blog.csdnimg.cn/img_convert/7f6c3fadd71a074e7a4192ff76d8f048.png)

上述变化，在线性代数里，我们可以用矩阵相乘的形式简洁的来表示：

$\begin{bmatrix}\frac{1}{\sqrt 2} & \frac{1}{\sqrt 2} \\ -\frac{1}{\sqrt 2} & \frac{1}{\sqrt 2} \end{bmatrix} \begin{bmatrix} 3 \\ 2\end{bmatrix} = \begin{bmatrix} \frac{5}{\sqrt 2} \\ - \frac{1}{\sqrt 2} \end{bmatrix}$

再稍微推广一下，假如我们有 $m$ 个二维向量，只要将二维向量按列排成一个两行 $m$ 列矩阵，然后用「基矩阵」乘以这个矩阵，就得到了所有这些向量在新基下的值。例如 $(1,1$) 、$(2,2$) 、$(3,3$)，想变换到刚才那组基上，可以如下这样表示：

$\begin{bmatrix}\frac{1}{\sqrt 2} & \frac{1}{\sqrt 2} \\ -\frac{1}{\sqrt 2} & \frac{1}{\sqrt 2} \end{bmatrix} \begin{bmatrix} 1 & 2 & 3 \\ 1 & 2 & 3\end{bmatrix} = \begin{bmatrix} 2\sqrt 2 & 4\sqrt2 & 6\sqrt2 \\ 0 & 0 & 0 \end{bmatrix}$

3.方差
----

在本文的开始部分，我们提到了，降维的目的是希望压缩数据但信息损失最少，也就是说，我们希望投影后的数据尽可能分散开。在数学上，这种分散程度我们用「方差」来表达，方差越大，数据越分散。

我们来看一个具体的例子。假设我们 $5$ 个样本数据，分别是 $x_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ 、 $x_2 = \begin{bmatrix} 1 \\ 3\end{bmatrix}$ 、 $x_3 = \begin{bmatrix} 2 \\ 3\end{bmatrix}$ 、 $x_4 = \begin{bmatrix} 4 \\ 4\end{bmatrix}$ 、 $x_5 = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$，将它们表示成矩阵形式：$X = \begin{bmatrix} 1 & 1 & 2 & 4 & 2 \\ 1 & 3 & 3 & 4 & 4 \end {bmatrix}$。

![Image 43: PCA降维算法; 方差;](https://img-blog.csdnimg.cn/img_convert/4cb1b71f8b7afd52b9fa2767ecb4f880.png)

为了后续处理方便，我们首先将每个字段内所有值都减去字段均值，其结果是将每个字段都变为均值为 $0$。

我们看上面的数据，设第一个特征为 $a$，第二个特征为 $b$，则某个样本可以写作 $x_i = \begin{bmatrix} a \\ b \end {bmatrix}$  
且特征 $a$ 的均值为2，特征 $b$ 的均值为 $3$。所以，变换后

$X = \begin{bmatrix} -1 & -1 & 0 & 2 & 0 \\ -2 & 0 & 0 & 1 & 1 \end{bmatrix}$

$Var(a ) = \frac{\sqrt 6} {5}$ $Var(b ) = \frac{\sqrt 6} {5}$

4.协方差
-----

协方差（`Covariance`）是衡量两个变量在偏离其各自均值时是否同步的一种指标。对于二维随机变量 $ x_i = \begin{bmatrix} a \\ b \end{bmatrix} $，除了分别考虑特征 $ a, b $ 的期望与方差，还需要刻画它们之间的相互关系。

协方差的定义为：

$$
Cov(a, b) = \frac{1}{m} \sum_{i = 1}^{m} (a_i - \bar{a})(b_i - \bar{b})
$$

或者概率论中写作：

$$
Cov(a, b) = \mathbb{E}[(a - \mathbb{E}[a])(b - \mathbb{E}[b])]
$$

- 当 $ Cov(a, b) > 0 $：两变量正相关
- 当 $ Cov(a, b) < 0 $：两变量负相关
- 当 $ Cov(a, b) = 0 $：两变量**线性无关**（但不一定独立）

特别地，方差是协方差的特例，即：

$$
Var(a) = Cov(a, a)
$$

5.协方差矩阵
-------

对于二维随机变量 $x_i = \begin{bmatrix} a \\ b \end {bmatrix}$，定义协方差矩阵：

$$
C = \begin{bmatrix} 
\text{Var}(a) & \text{Cov}(a, b) \\ 
\text{Cov}(b, a) & \text{Var}(b)
\end{bmatrix}
$$

对于 $n$ 维随机变量：

$$
x_{i}=\left[\begin{array}{c} x_{1} \\ x_{2} \\ \vdots \\ x_{n} \end{array}\right]
$$

协方差矩阵定义为：

$$
C = \begin{bmatrix} 
\text{Var}(x_1) & \text{Cov}(x_1, x_2) &\cdots & \text{Cov}(x_1, x_n)\\ 
\text{Cov}(x_2, x_1) & \text{Var}(x_2) & \cdots & \text{Cov}(x_2, x_n)\\ 
\vdots & \vdots & \ddots & \vdots \\ 
\text{Cov}(x_n, x_1) & \text{Cov}(x_n, x_2) & \cdots & \text{Var}(x_n) 
\end{bmatrix}
$$

协方差矩阵是一个 $n \times n$ 的对称矩阵，其对称性来源于协方差的性质 $\text{Cov}(X,Y) = \text{Cov}(Y,X)$。主对角线上的元素是各维度的方差，非对角线元素是两两维度之间的协方差。

如果有 $ m $ 个 $ n $ 维样本，将每个样本表示为列向量 $ x^{(i)} \in \mathbb{R}^n $，我们将它们按列排成一个矩阵 $ X \in \mathbb{R}^{n \times m} $：

$$
X = \begin{bmatrix} 
\vert & \vert &        & \vert \\
x^{(1)} & x^{(2)} & \cdots & x^{(m)} \\
\vert & \vert &        & \vert
\end{bmatrix}
$$

例如对于二维样本 $ (a_i, b_i) $，每个样本为一个二维列向量，矩阵 $ X \in \mathbb{R}^{2 \times m} $ 表示为：

$$
X = \begin{bmatrix} 
a_1 & a_2 & \cdots & a_m \\ 
b_1 & b_2 & \cdots & b_m 
\end{bmatrix}
$$

对去中心化后的矩阵 $X$ 进行变换，计算 $X$ 乘以其转置 $X^T$，并乘上系数 $1/m$：

$$
\frac{1}{m} X X^T = \frac{1}{m} \begin{bmatrix} 
a_1 & a_2 & \cdots & a_m \\ 
b_1 & b_2 & \cdots & b_m 
\end{bmatrix}
\begin{bmatrix} 
a_1 & b_1 \\ 
a_2 & b_2 \\ 
\vdots & \vdots \\ 
a_m & b_m 
\end{bmatrix}
= \begin{bmatrix} 
\frac{1}{m} \sum_{i=1}^m a_i^2 & \frac{1}{m} \sum_{i=1}^m a_i b_i \\ 
\frac{1}{m} \sum_{i=1}^m a_i b_i & \frac{1}{m} \sum_{i=1}^m b_i^2 
\end{bmatrix}
$$

> 注：右侧第二个矩阵为 $X^T$，即将原样本矩阵的列向量转置为行向量所形成的 $m \times n$ 矩阵。

这正是协方差矩阵！我们归纳得到：设我们有 $m$ 个 $n$ 维去中心化后的数据记录，将其按列排成 $n \times m$ 的矩阵 $X$，则协方差矩阵 $C$ 可以表示为：

$$
C = \frac{1}{m} X X^T
$$

协方差矩阵是一个对称矩阵，其对角线元素表示各个特征的方差，非对角线元素表示两两特征之间的协方差。特别地，第 $i$ 行 $j$ 列和第 $j$ 行 $i$ 列的元素相同，均为特征 $i$ 和特征 $j$ 之间的协方差。

### 示例

我们有以下 5 个二维样本数据：

$$
x_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix},\ 
x_2 = \begin{bmatrix} 1 \\ 3 \end{bmatrix},\ 
x_3 = \begin{bmatrix} 2 \\ 3 \end{bmatrix},\ 
x_4 = \begin{bmatrix} 4 \\ 4 \end{bmatrix},\ 
x_5 = \begin{bmatrix} 2 \\ 4 \end{bmatrix}
$$

首先计算每个维度的均值：

- 第一个维度的均值：$\mu_1 = \frac{1+1+2+4+2}{5} = 2$
- 第二个维度的均值：$\mu_2 = \frac{1+3+3+4+4}{5} = 3$

去中心化后的数据矩阵（每列是样本，每行是维度）：

$$
X = \begin{bmatrix} 
-1 & -1 & 0 & 2 & 0 \\ 
-2 & 0 & 0 & 1 & 1 
\end{bmatrix}
$$

计算协方差矩阵：

$$
C = \frac{1}{5} X X^T
$$

首先计算 $ X X^T $：

$$
X X^T = \begin{bmatrix} 
-1 & -1 & 0 & 2 & 0 \\ 
-2 & 0 & 0 & 1 & 1 
\end{bmatrix}
\begin{bmatrix} 
-1 & -2 \\ 
-1 & 0 \\ 
0 & 0 \\ 
2 & 1 \\ 
0 & 1 
\end{bmatrix}
= \begin{bmatrix} 
(-1)(-1) + (-1)(-1) + 0 \cdot 0 + 2 \cdot 2 + 0 \cdot 0 & (-1)(-2) + (-1) \cdot 0 + 0 \cdot 0 + 2 \cdot 1 + 0 \cdot 1 \\ 
(-2)(-1) + 0 \cdot (-1) + 0 \cdot 0 + 1 \cdot 2 + 1 \cdot 0 & (-2)(-2) + 0 \cdot 0 + 0 \cdot 0 + 1 \cdot 1 + 1 \cdot 1 
\end{bmatrix}
$$

因此：

$$
X X^T = \begin{bmatrix} 
6 & 4 \\ 
4 & 6 
\end{bmatrix}
$$

乘以系数 $ \frac{1}{5} $：

$$
C = \frac{1}{5} \begin{bmatrix} 
6 & 4 \\ 
4 & 6 
\end{bmatrix} = \begin{bmatrix} 
\frac{6}{5} & \frac{4}{5} \\ 
\frac{4}{5} & \frac{6}{5} 
\end{bmatrix}
$$

所以，示例数据的协方差矩阵为：

$$
C = \begin{bmatrix} 
1.2 & 0.8 \\ 
0.8 & 1.2 
\end{bmatrix}
$$

6.协方差矩阵对角化
----------

再回到我们的场景和目标：

设 $X$ 的协方差矩阵为 $C$，$Y$ 的协方差矩阵为 $D$，且 $Y = PX$。

那么 $C$ 与 $D$ 是什么关系呢？

$\begin{aligned}  D & =\frac{1}{m} Y Y^{T} \\  & =\frac{1}{m}(P X)(P X)^{T} \\  & =\frac{1}{m} P X X^{T} P^{T} \\  & =\frac{1}{m} P\left(X X^{T}\right) P^{T} \\  & =P C P^{T} \\  & =P\left[\begin{array}{cc} \frac{1}{m} \sum_{i=1}^{m} a_{i}^{2} & \frac{1}{m} \sum_{i=1}^{m} a_{i} b_{i} \\ \frac{1}{m} \sum_{i=1}^{m} a_{i} b_{i} & \frac{1}{m} \sum_{i=1}^{m} b_{i}^{2} \end{array}\right] P^{T} \end{aligned}$

我们发现，要找的 $P$ 不是别的，而是能让原始协方差矩阵对角化的 $P$。

换句话说，优化目标变成了寻找一个矩阵 $P$，满足 $PCP^T$ 是一个对角矩阵，并且对角元素按从大到小依次排列，那么 $P$ 的前 $K$ 行就是要寻找的基，用 $P$ 的前 $K$ 行组成的矩阵乘以 $X$ 就使得 $X$ 从 $N$ 维降到了 $K$ 维并满足上述优化条件。

最终我们聚焦在协方差矩阵对角化这个问题上。

由上文知道，协方差矩阵 $C$ 是一个是对称矩阵，在线性代数上，实对称矩阵有一系列非常好的性质：

* 1）实对称矩阵不同特征值对应的特征向量必然正交。
* 2）设特征向量 $\lambda$ 重数为 $r$，则必然存在 $r$ 个线性无关的特征向量对应于 $\lambda$，因此可以将这 $r$ 个特征向量单位正交化。

由上面两条可知，一个 $n$ 行 $n$ 列的实对称矩阵一定可以找到 $n$ 个单位正交特征向量，设这 $n$ 个特征向量为 $e_1,e_2,⋯,e_n$，我们将其按列组成矩阵：

$E = \begin{bmatrix} e_1 & e_2 & \cdots \ e_n\end{bmatrix}$

则对协方差矩阵 $C$ 有如下结论：

$E^T C E = \Lambda = \begin{bmatrix} \lambda_1 \\ & \lambda_2 \\ &&\ddots \\ &&&\lambda_n\end {bmatrix}$

其中 $\Lambda$ 为对角矩阵，其对角元素为各特征向量对应的特征值（可能有重复）。  
结合上面的公式：

$D = PCP^T$

其中，$D$ 为对角矩阵，我们可以得到：

$P = E^T$

$P$ 是协方差矩阵$C$的特征向量单位化后按行排列出的矩阵，其中每一行都是 $C$ 的一个特征向量。如果设 $P$ 按照 $\Lambda$ 中特征值的从大到小，将特征向量从上到下排列，则用 $P$ 的前 $K$ 行组成的矩阵乘以原始数据矩阵 $X$，就得到了我们需要的降维后的数据矩阵 $Y$。

7.PCA 算法
-------

总结一下 PCA 的算法步骤：

![Image 143: PCA降维算法; PCA的算法步骤;](https://img-blog.csdnimg.cn/img_convert/c6a0b8f2e9a35fb07a40cd82879b3075.png)

设有 $m$ 条 $n$ 维数据。

1）将原始数据按列组成 $n$ 行 $m$ 列矩阵 $X$

2）将 $X$ 的每一行（代表一个特征）进行零均值化，即减去这一行的均值

3）求出协方差矩阵 $C=\frac{1}{m}XX^T$

4）求出协方差矩阵 $C$ 的特征值及对应的特征向量

5）将特征向量按对应特征值大小从上到下按行排列成矩阵，取前 $k$ 行组成矩阵 $P$

6） $Y=PX$ 即为降维到 $k$ 维后的数据

8. PCA 代码实践
---------

我们这里直接使用 python 机器学习工具库 scikit-learn 来给大家演示PCA算法应用（相关知识速查可以查看ShowMeAI文章 [AI建模工具速查|Scikit-learn使用指南](https://www.showmeai.tech/article-detail/108)），sklearn 工具库中与 PCA 相关的类都在 `sklearn.decomposition` 包里，最常用的 PCA 类就是 `sklearn.decomposition.PCA`。

### 1）参数介绍

sklearn 中的 PCA 类使用简单，基本无需调参，一般只需要指定需要降维到的维度，或者降维后的主成分的方差和占原始维度所有特征方差和的比例阈值就可以了。

下面是**sklearn.decomposition.PCA**的主要参数介绍：

*   **n\_components**：PCA 降维后的特征维度数目。
    
*   **whiten**：是否进行白化。所谓白化，就是对降维后的数据的每个特征进行归一化，让方差都为 $1$，默认值是False，即不进行白化。
    
*   **svd\_solver**：奇异值分解 SVD 的方法，有 $4$ 个可以选择的值：{`auto`,`full`,`arpack`,`randomized`}。
    

除上述输入参数，还有两个 PCA 类的成员属性也很重要：

* **explained_variance_**，它代表降维后的各主成分的方差值。
    
* **explained_variance\_ratio_**，它代表降维后的各主成分的方差值占总方差值的比例。
    

### 2）代码实例

```python
## 构建数据样本并可视化
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
from sklearn.datasets import make_blobs
## X为样本特征，Y为样本簇类别， 共1000个样本，每个样本3个特征，共4个簇
X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3,3, 3], [0,0,0], [1,1,1], [2,2,2]], cluster_std=[0.2, 0.1, 0.2, 0.2], 
                  random_state =9)
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
plt.scatter(X[:, 0], X[:, 1], X[:, 2],marker='x')

```

![Image 158: PCA降维算法; PCA代码实践;](https://img-blog.csdnimg.cn/img_convert/e3510d5b028e06cd546fe5224a0794d8.png)

先不降维，只对数据进行投影，看看投影后的三个维度的方差分布，代码如下：

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

```

输出如下：

```text
[0.983182120.008500370.00831751]
[3.785216380.032726130.03202212]
```

可以看出投影后三个特征维度的方差比例大约为 $98.3\%:0.8\%:0.8\%$。投影后第一个特征占了绝大多数的主成分比例。现在我们来进行降维，从三维降到 $2$ 维，代码如下：

```python
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
```

输出如下：

```text
[0.983182120.00850037]
[3.785216380.03272613]
```

这个结果其实可以预料，因为上面三个投影后的特征维度的方差分别为：$[3.78521638 \quad 0.03272613]$，投影到二维后选择的肯定是前两个特征，而抛弃第三个特征。为了有个直观的认识，我们看看此时转化后的数据分布，代码如下：

```python
X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1],marker='x')
plt.show()
```

![Image 162: PCA降维算法; PCA代码实践;](https://img-blog.csdnimg.cn/img_convert/e7836c7bd57d8e6d221a6ae6da07b816.png)

从上图可以看出，降维后的数据依然清楚可见之前三维图中的 $4$ 个簇。现在我们不直接指定降维的维度，而指定降维后的主成分方差和比例，来试验一下。

```python
pca = PCA(n_components=0.9)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)
```

我们指定了主成分至少占90% $ \%$，输出如下：

```text
[0.98318212]
[3.78521638]
1
```
可见只有第一个投影特征被保留。这也很好理解，我们的第一个主成分占投影特征的方差比例高达 $98 \%$。只选择这一个特征维度便可以满足 $90 \%$ 的阈值。我们现在选择阈值 $99 \%$ 看看，代码如下：

```python
pca = PCA(n_components=0.99)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)
```

此时的输出如下：

```text
[0.983182120.00850037]
[3.785216380.03272613]
2
```

这个结果也很好理解，因为我们第一个主成分占了 $98.3 \%$ 的方差比例，第二个主成分占了 $0.8 \%$ 的方差比例，两者一起可以满足我们的阈值。最后我们看看让MLE算法自己选择降维维度的效果，代码如下：

```python
pca = PCA(n_components= 'mle',svd_solver='full')
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)
```

输出结果如下：

```text
[0.98318212]
[3.78521638]
1
```

可见由于我们的数据的第一个投影特征的方差占比高达 $98.3 \%$，MLE 算法只保留了我们的第一个特征。