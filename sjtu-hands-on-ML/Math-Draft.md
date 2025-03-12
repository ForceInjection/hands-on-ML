### **概率分布**  
联合分布分解为似然和先验的乘积：  

$$
p(\mathbf{y}, \mathbf{w}|\mathbf{X}) = p(\mathbf{y}|\mathbf{w}, \mathbf{X}) p(\mathbf{w}) = p(\mathbf{w}) \prod_{i=1}^{N} p(y_i|\mathbf{w}, \mathbf{x}_i)
$$

### **贝叶斯公式与后验分布**  
根据贝叶斯定理，模型参数的后验分布为：  

$$
p(\mathbf{w}|\mathbf{X}, \mathbf{y}) = \frac{p(\mathbf{y}|\mathbf{w}, \mathbf{X}) p(\mathbf{w})}{p(\mathbf{y}|\mathbf{X})} \propto p(\mathbf{w}) \prod_{i=1}^{N} p(y_i|\mathbf{w}, \mathbf{x}_i)
$$

### **先验分布与似然函数**  
1. **先验分布**（拉普拉斯分布，对应 L1 正则项）：  
   
   $$
   p(\mathbf{w}|\mu=0, b) = \frac{1}{2b} \exp\left(-\frac{\|\mathbf{w}\|_1}{b}\right), \quad \mathbf{w} \sim \text{Laplace}(0, b)
   $$

2. **似然函数**（高斯分布，对应平方误差项）：  
   
   $$
   y_i \sim \mathcal{N}(\mathbf{x}_i^T \mathbf{w}, \sigma^2), \quad p(y_i|\mathbf{x}_i, \mathbf{w}, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \mathbf{w}^T\mathbf{x}_i)^2}{2\sigma^2}\right)
   $$

### **后验分布推导**  
后验分布的表达式为：  

$$
\begin{aligned}
p(\mathbf{w}|\mathbf{X}, \mathbf{y}) 
&\propto p(\mathbf{w}|\mu=0, b) \prod_{i=1}^{N} \mathcal{N}(y_i|\mathbf{x}_i^T \mathbf{w}, \sigma^2) \\
&= \frac{1}{2b} \exp\left(-\frac{\|\mathbf{w}\|_1}{b}\right) \prod_{i=1}^{N} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \mathbf{w}^T\mathbf{x}_i)^2}{2\sigma^2}\right)
\end{aligned}
$$

### **最大后验估计（MAP）**  
目标函数取对数并简化：  

$$
\begin{aligned}
\arg\max_{\mathbf{w}} p(\mathbf{w}|\mathbf{X}, \mathbf{y}) 
&= \arg\max_{\mathbf{w}} \left[ \log p(\mathbf{w}) + \sum_{i=1}^{N} \log p(y_i|\mathbf{w}, \mathbf{x}_i) \right] \\
&= \arg\max_{\mathbf{w}} \left[ -\frac{\|\mathbf{w}\|_1}{b} - \frac{1}{2\sigma^2} \sum_{i=1}^{N} (y_i - \mathbf{w}^T\mathbf{x}_i)^2 \right] \\
&= \arg\min_{\mathbf{w}} \left( \frac{\|\mathbf{w}\|_1}{b} + \frac{1}{2\sigma^2} \sum_{i=1}^{N} (y_i - \mathbf{w}^T\mathbf{x}_i)^2 \right)
\end{aligned}
$$

### **最终优化问题**  

$$
\min_{\mathbf{w}} \left( \underbrace{\frac{\|\mathbf{w}\|_1}{b}}_{\text{L1 正则项}} + \underbrace{\frac{1}{2\sigma^2} \sum_{i=1}^{N} (y_i - \mathbf{w}^T\mathbf{x}_i)^2}_{\text{平方误差项}} \right)
$$
