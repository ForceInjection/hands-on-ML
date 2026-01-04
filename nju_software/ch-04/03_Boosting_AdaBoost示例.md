# AdaBoost 算法迭代过程示例

本文档旨在通过一个包含 10 个样本的直观示例，详细拆解 AdaBoost 算法的完整迭代流程。我们将从零开始，手动推导每一轮弱分类器的选择逻辑、系数计算以及样本权重的更新细节，并最终验证强分类器的预测效果。

文档结构包含三部分：

1. **示例题目**：定义数据集与候选弱分类器。
2. **手工推导**：逐步展示三轮迭代的数值计算过程，直观呈现权重分布的变化。
3. **代码复现**：提供可直接运行的 Python 代码，用于验证手算结果并辅助教学演示。

通过本案例，读者将深入理解 AdaBoost 如何通过“关注难分样本”的机制，将多个简单的弱分类器组合成一个强大的强分类器。

## 一、示例题目

### 1.1 数据集

共有 10 个样本，特征为一维实数 $x_i$，标签 $y_i \in \{+1,-1\}$。下表按样本序号 $i=1,\dots,10$ 列出：

| 样本序号 $i$ | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   | 10  |
| ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $x_i$        | 0   | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   |
| $y_i$        | 1   | 1   | 1   | -1  | -1  | -1  | 1   | 1   | 1   | -1  |

### 1.2 候选弱分类器

设弱分类器输出为 $\{-1,+1\}$，定义 3 个候选弱分类器：

- $G_1(x)=\begin{cases}1,&x<2.5\\-1,&\text{otherwise}\end{cases}$
- $G_2(x)=\begin{cases}1,&x<8.5\\-1,&\text{otherwise}\end{cases}$
- $G_3(x)=\begin{cases}1,&x\ge 5.5\\-1,&\text{otherwise}\end{cases}$

### 1.3 任务

模拟 AdaBoost（二分类，指数损失版本）在该数据集上的训练过程，进行 3 轮迭代；每一轮从 $\{G_1,G_2,G_3\}$ 中选择**加权分类误差**最小的弱分类器并更新样本权重。

---

## 二、手工推导

### 2.1 符号与公式

设第 $t$ 轮的样本分布为 $D_t(i)$（$i$ 为样本序号），弱分类器为 $G_t$，其加权误差为：

$$
\varepsilon_t=\sum_{i=1}^{n} D_t(i)\,\mathbb{I}\big(G_t(x_i)\ne y_i\big)
$$

弱分类器系数（投票权重）为：

$$
\alpha_t=\frac{1}{2}\ln\left(\frac{1-\varepsilon_t}{\varepsilon_t}\right)
$$

样本权重更新（并归一化使其和为 1）：

$$
D_{t+1}(i)=\frac{D_t(i)\exp\big(-\alpha_t y_i G_t(x_i)\big)}{Z_t},
\quad
Z_t=\sum_{k=1}^{n} D_t(k)\exp\big(-\alpha_t y_k G_t(x_k)\big)
$$

### 2.2 初始化

样本数 $n=10$，初始分布为均匀分布：

$$
D_1(i)=\frac{1}{10}=0.1,\quad i=1,\dots,10
$$

### 2.3 第 1 轮：在 $\{G_1,G_2,G_3\}$ 中选择误差最小者

先列出每个候选弱分类器的预测与错误样本（此轮 $D_1$ 均匀，因此加权误差等于错误样本比例）：

- $G_1$ 错误：$x=6,7,8$，$\varepsilon(G_1)=0.1+0.1+0.1=0.3$
- $G_2$ 错误：$x=3,4,5$，$\varepsilon(G_2)=0.3$
- $G_3$ 错误：$x=0,1,2,9$，$\varepsilon(G_3)=0.4$

取误差最小者。此处 $G_1$ 与 $G_2$ 并列最优（均为 0.3），为便于复现，约定“并列时取编号更小者”，因此本轮选 $G_1$。

**(1) 计算误差与系数**：

$$
\varepsilon_1=0.3,
\quad
\alpha_1=\frac{1}{2}\ln\left(\frac{1-0.3}{0.3}\right)\approx 0.4236
$$

**(2) 更新权重并归一化**：

本轮 $G_1$ 错误样本是 $x=6,7,8$（3 个），正确样本是其余 7 个。更新前 $D_1(i)=0.1$：

- 正确样本：$D_2(i)\propto 0.1\cdot e^{-\alpha_1}$
- 错误样本：$D_2(i)\propto 0.1\cdot e^{+\alpha_1}$

归一化后（四舍五入到 4 位小数）：

$$
D_2\approx[0.0714,0.0714,0.0714,0.0714,0.0714,0.0714,0.1667,0.1667,0.1667,0.0714]
$$

### 2.4 第 2 轮：重新选择弱分类器

在分布 $D_2$ 下计算候选误差：

- $G_1$ 错误：$x=6,7,8$，$\varepsilon(G_1)=0.1667\times 3=0.5$
- $G_2$ 错误：$x=3,4,5$，$\varepsilon(G_2)=0.0714\times 3\approx 0.2143$
- $G_3$ 错误：$x=0,1,2,9$，$\varepsilon(G_3)=0.0714\times 3+0.0714\approx 0.2857$

本轮选择 $G_2$。

**(1) 计算误差与系数**：

$$
\varepsilon_2\approx 0.2143,
\quad
\alpha_2=\frac{1}{2}\ln\left(\frac{1-\varepsilon_2}{\varepsilon_2}\right)\approx 0.6496
$$

**(2) 更新权重并归一化**：

$G_2$ 的错误样本为 $x=3,4,5$，归一化后：

$$
D_3\approx[0.0455,0.0455,0.0455,0.1667,0.1667,0.1667,0.1061,0.1061,0.1061,0.0455]
$$

### 2.5 第 3 轮：重新选择弱分类器

在分布 $D_3$ 下计算候选误差：

- $G_1$ 错误：$x=6,7,8$，$\varepsilon(G_1)=0.1061\times 3\approx 0.3183$
- $G_2$ 错误：$x=3,4,5$，$\varepsilon(G_2)=0.1667\times 3=0.5$
- $G_3$ 错误：$x=0,1,2,9$，$\varepsilon(G_3)=0.0455\times 4\approx 0.1818$

本轮选择 $G_3$。

**(1) 计算误差与系数**：

$$
\varepsilon_3\approx 0.1818,
\quad
\alpha_3=\frac{1}{2}\ln\left(\frac{1-\varepsilon_3}{\varepsilon_3}\right)\approx 0.7520
$$

**(2) 更新权重并归一化**：

$$
D_4\approx[0.1250,0.1250,0.1250,0.1019,0.1019,0.1019,0.0648,0.0648,0.0648,0.1250]
$$

---

## 三、结果汇总与检验

### 3.1 三轮选择结果

| 轮次 $t$ | 选择的弱分类器 $G_t$ | 加权误差 $\varepsilon_t$ | 系数 $\alpha_t$ |
| -------- | -------------------- | ------------------------ | --------------- |
| 1        | $G_1$                | 0.3000                   | 0.4236          |
| 2        | $G_2$                | 0.2143                   | 0.6496          |
| 3        | $G_3$                | 0.1818                   | 0.7520          |

### 3.2 最终强分类器

强分类器为加权投票：

$$
F(x)=\operatorname{sign}\Big(\alpha_1G_1(x)+\alpha_2G_2(x)+\alpha_3G_3(x)\Big)
$$

代入数值：

$$
F(x)=\operatorname{sign}\left(0.4236\cdot G_1(x)+0.6496\cdot G_2(x)+0.7520\cdot G_3(x)\right)
$$

### 3.3 训练集上的分类结果

将 $x=0,1,\dots,9$ 代入，可得到 $F(x_i)=y_i$（该 10 个训练样本全部分类正确），训练误差为 0。

---

## 四、Python 代码实现

下面代码实现“每轮从候选集合中选择加权误差最小者”的 AdaBoost，并按文中并列规则（误差相同取编号小者）选出 $G_1\rightarrow G_2\rightarrow G_3$。代码中包含必要注释，便于课堂讲解与学生复现。

```python
import numpy as np

# 题目给定数据集（x 为一维特征，y 为 {-1, +1} 标签）
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1], dtype=float)
n = y.size

# 候选弱分类器（向量化实现，返回形状为 (n,) 的预测向量）
def G1(x_vec: np.ndarray) -> np.ndarray:
    # 若 x < 2.5 则输出 +1，否则输出 -1
    return np.where(x_vec < 2.5, 1.0, -1.0)

def G2(x_vec: np.ndarray) -> np.ndarray:
    # 若 x < 8.5 则输出 +1，否则输出 -1
    return np.where(x_vec < 8.5, 1.0, -1.0)

def G3(x_vec: np.ndarray) -> np.ndarray:
    # 若 x >= 5.5 则输出 +1，否则输出 -1
    return np.where(x_vec >= 5.5, 1.0, -1.0)

candidates = [("G1", G1), ("G2", G2), ("G3", G3)]

# 初始化样本权重分布 D1（均匀分布）
D = np.full(shape=n, fill_value=1.0 / n, dtype=float)

chosen = []
alphas = []

for t in range(1, 4):
    # 1) 在候选集合中选出加权误差最小的弱分类器（并列取列表中更靠前者）
    best_name = None
    best_pred = None
    best_error = None
    best_idx = None

    for idx, (name, G) in enumerate(candidates):
        pred = G(x)
        error = float(np.sum(D[pred != y]))
        if best_error is None or error < best_error - 1e-12 or (abs(error - best_error) <= 1e-12 and idx < best_idx):
            best_name = name
            best_pred = pred
            best_error = error
            best_idx = idx

    # 2) 计算 alpha（为数值稳定性，避免误差为 0 或 1）
    eps = 1e-12
    clipped_error = min(max(best_error, eps), 1.0 - eps)
    alpha = 0.5 * np.log((1.0 - clipped_error) / clipped_error)

    # 3) 更新权重并归一化：D_{t+1}(i) ∝ D_t(i) * exp(-alpha * y_i * pred_i)
    D = D * np.exp(-alpha * y * best_pred)
    D = D / np.sum(D)

    chosen.append(best_name)
    alphas.append(alpha)

    # 4) 输出便于核对的中间结果
    print(f"第 {t} 轮选择: {best_name}, 误差: {best_error:.4f}, alpha: {alpha:.4f}")
    print("更新后的 D:", np.round(D, 4))
    print("-" * 60)

# 最终强分类器在训练集上的预测（sign(Σ alpha_t G_t(x))）
score = np.zeros_like(y)
for name, alpha in zip(chosen, alphas):
    G = dict(candidates)[name]
    score += alpha * G(x)

y_pred = np.where(score >= 0, 1.0, -1.0)
train_error = float(np.mean(y_pred != y))

print("最终选择顺序:", chosen)
print("最终 alpha:", [round(float(a), 4) for a in alphas])
print("训练误差:", train_error)
```

---

## 五、总结

本示例展示了 AdaBoost 算法“化弱为强”的核心机制：通过迭代调整样本权重，迫使算法在每一轮中关注上一轮被分错的“难样本”。在三轮迭代中，我们观察到权重分布 $D_t$ 逐渐向分类错误的边界样本（如 $x=3,4,5$ 或 $x=0,1,2,9$）集中，从而引导后续的弱分类器去修补之前的错误。

---

## 六、参考文献

- Yoav Freund, Robert E. Schapire. “A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting”. Journal of Computer and System Sciences, 55(1), 1997.
- 李航，《统计学习方法》（第 2 版），清华大学出版社，2019。第 8 章 AdaBoost 算法（推导公式与符号约定）。
