---
title: Weighted Logistic Regression 调查报告
date: 2026-08-31 21:00:00
categories:
  - 推荐系统
tags:
  - Weighted LR
  - YouTube
  - 时长建模
  - 推荐系统
---

# Weighted Logistic Regression 

> 调查主题：YouTube 推荐系统中 Weighted LR 的原理、推导、Serving 方式与应用
---

## 一、问题背景

YouTube 深度学习推荐系统的 Ranking Model 输出层采用了 **Weighted Logistic Regression**，且在模型 Serving 时没有使用标准的 sigmoid 函数，而是使用了 $e^{Wx+b}$ 这一指数形式。这个设计有几个绕不开的问题：为什么用 Weighted LR？为什么 Serving 时用指数形式而非 sigmoid？预测的到底是什么？


---

## 二、LR 基础推导

逻辑回归假设数据服从**伯努利分布**，事件发生概率为 $p$，不发生概率为 $1-p$。

### 2.1 Odds（几率比）

$$Odds = \frac{p}{1-p}$$

Odds 表示事件发生与不发生的比值。

### 2.2 Logit 函数与 LR 的由来

对 Odds 取自然对数，并令其等于线性回归函数：

$$\ln\left(\frac{p}{1-p}\right) = \theta^T x \quad \Rightarrow \quad \frac{p}{1-p} = e^{\theta^T x}$$

$$\Rightarrow p = \frac{1}{1+e^{-\theta^T x}} = \text{sigmoid}(\theta^T x)$$

这就是逻辑回归的由来：**sigmoid 函数的选择不是任意的，而是由伯努利分布 + logit 链接函数推导而来**。

### 2.3 关键转换

$$\ln(Odds) = \theta^T x \quad \Rightarrow \quad Odds = e^{\theta^T x}$$

> 这一步是理解 YouTube Serving 函数的关键：$e^{Wx+b}$ 算的是 Odds，不是概率 $p$。


---

## 三、Weighted LR 的核心思想

### 3.1 权重设计

YouTube 论文中的权重设计：

$$w(x, y) = \begin{cases} T_i \text{（观看时长）} & \text{当 } y = 1 \text{（正样本，点击）} \\ 1 & \text{当 } y = 0 \text{（负样本，未点击）} \end{cases}$$


### 3.2 加权后 Odds 的变化

直觉上，把正样本的权重乘上 $w_i$，等价于**把这个正样本复制 $w_i$ 份**再训练标准 LR。复制之后，正类的等效质量从 $p$ 膨胀为 $w_i \cdot p$，而负类质量不变（负样本权重为 1）。加权后的几率比因此变为：

$$Odds_w = \frac{w_i \cdot p}{1 - p}$$

其中 $p$ 为用户打开视频的概率（即 pCTR），$w_i = T_i$ 为观看时长。

> 一个常见的写法混淆：不少资料把加权后的 odds 写成 $\frac{w_i p}{1 - w_i p}$，即把 $w_i p$ 当作新的概率再算 odds。这个写法隐含 $w_i p < 1$ 的约束——时长较长时（如 $T_i = 100$ 秒、$p = 0.01$，此时 $w_i p = 1$）分母直接归零，得出荒谬结果。从加权似然严格推导（见 4.2 与 4.4 节）得到的分母是 $1-p$，与时长无关，不存在这个问题。

### 3.3 关键近似：pCTR 很小时

在视频推荐场景中，用户打开视频的概率 $p$ 往往是一个很小的值，分母 $1-p \approx 1$，因此：

$$Odds_w = \frac{w_i \cdot p}{1 - p} \approx w_i \cdot p = T_i \cdot p = E(T_i)$$

到这里就清楚了：$T_i \cdot p$ 就是用户观看某视频的期望时长。

注意近似条件是 **$p$ 很小**而非 $w_i p$ 很小——即使个别样本时长很长，只要整体 pCTR 低，近似依然成立。这也解释了 6.1 节数值实验中 pCTR 高达 50% 时估计严重偏离的根源：分母 $1-p$ 已经不可忽略。


---

## 四、形式化数学推导

### 4.1 从加权数据分布出发

引入采样权重后，训练数据服从新的分布 $q$：

$$q(x, y) = \frac{w(x, y) \cdot p(x, y)}{Z}$$

其中 $Z$ 为归一化常数。注意 $Z$ 不影响任何结论的真正原因：我们关心的都是**条件几率比** $\frac{q(y=1 \mid x)}{q(y=0 \mid x)}$，分子分母同除 $Z$ 后它被约掉——条件分布根本不依赖 $Z$。

### 4.2 加权后的个体预测

加权后，个体级别的 log-odds 变为：

$$\frac{q(y=1|x)}{q(y=0|x)} = \frac{w(x,1) \cdot p(y=1|x)}{p(y=0|x)}$$

这意味着模型学习的是 **log(观看时长 × pCTR)**：

$$e^{x^T \beta} = \frac{w(x,1) \cdot p(y=1|x)}{p(y=0|x)} \approx w(x,1) \cdot p(y=1|x)$$

即模型预测的是**单个用户的期望观看时长**。

### 4.3 全局 Odds 与期望观看时长的关系

$$\frac{q(y=1)}{q(y=0)} = \frac{\sum_x w(x,1) \cdot p(y=1|x) \cdot p(x)}{p(y=0)} = \frac{1}{N} \sum_{\text{users}} (\text{expected watch time}) = E(w)$$

当 $p(x) \approx 1/N$（均匀分布假设）时，全局 Odds 等于**所有用户的期望观看时长之和除以 N**，即期望观看时间。

### 4.4 严格推导——WLR 到底估计的是什么

前面几节的结论（$e^{w\cdot x} \approx E(T)$）都建立在近似上。这一节从加权似然出发做严格推导，回答一个更精确的问题：**WLR 的估计是观看时长的均值吗？**

**设定**：$p(x) = P(y{=}1 \mid x)$ 为点击概率，$T$ 为观看时长（未点击约定 $T=0$），期望观看时长 $m(x) = E[T \mid x] = p(x) \cdot E[T \mid y{=}1, x]$。

**第 1 步：写出加权目标的总体形式。** WLR 的损失对全体样本取期望，正样本项的权重期望化后恰好是 $m(x)$：

$$\min_\theta \; E_x\Big[ m(x) \cdot \big(-\ln \pi(x)\big) + \big(1 - p(x)\big)\cdot\big(-\ln(1-\pi(x))\big) \Big]$$

其中 $\pi(x) = \sigma(w \cdot x)$。直觉：正类的等效质量被时长撑大为 $m(x)$，负类质量保持 $1-p(x)$。

**第 2 步：逐点求导置零。** 目标对每个 $x$ 的 $\pi(x)$ 独立可分，逐点优化：

$$\frac{\partial}{\partial \pi} = -\frac{m(x)}{\pi} + \frac{1-p(x)}{1-\pi} = 0 \implies m(x)(1-\pi) = \pi\big(1-p(x)\big)$$

**第 3 步：解出最优解。** 对一阶条件解关于 $\pi$ 的线性方程（交叉相乘、移项合并即得）：

$$\pi^*(x) = \frac{m(x)}{m(x) + 1 - p(x)}$$

**第 4 步：从 $\pi^*$ 到 $e^{w\cdot x}$——用 sigmoid 的 odds 恒等式。** 模型输出 $\pi(x) = \sigma(w\cdot x)$，而 sigmoid 有一个核心恒等式（正是 2.3 节的关键转换）：**其 odds 恒等于输入的指数**：

$$\frac{\sigma(z)}{1-\sigma(z)} = \frac{e^z/(1+e^z)}{1/(1+e^z)} = e^z$$

因此模型收敛时 $e^{w\cdot x} = \frac{\pi^*}{1-\pi^*}$。先算出分母：

$$1 - \pi^* = \frac{(m+1-p) - m}{m+1-p} = \frac{1-p}{m+1-p}$$

再做分数相除（除以分数等于乘其倒数），公共分母 $m+1-p$ 上下相消：

$$e^{w\cdot x} = \frac{\dfrac{m}{m+1-p}}{\dfrac{1-p}{m+1-p}} = \frac{m(x)}{1-p(x)} = \frac{E[T \mid x]}{1 - p(x)}$$

**数值例子**：设 $p = 0.05$、$m = E[T \mid x] = 4$ 分钟。加权后 $\pi^* = \frac{4}{4.95} \approx 0.81$——注意它已不是点击概率（正类等效质量被时长撑大了 4 倍）；odds $= \frac{0.81}{0.19} \approx 4.21$；用公式直接验证 $\frac{4}{0.95} \approx 4.21$，一致。低 pCTR 近似下 $4.21 \approx 4 = E[T \mid x]$，偏高恰好 $\frac{1}{1-p} \approx 1.053$ 倍——5% 的点击率带来 5.3% 的高估。

**结论**：WLR 的精确估计不是均值本身，而是 $\frac{E[T \mid x]}{1-p(x)}$，期望时长除以不点击概率。低 pCTR 场景（$p \ll 1$）下分母趋于 1，才退化为期望观看时长；pCTR = 20% 的场景会系统性高估 25%。

**两个检验**：

- **边界自洽**：若所有点击时长恒为 1（$T \equiv 1$），则 $m(x) = p(x)$，$\pi^* = \frac{p}{p + 1 - p} = p$——精确退化为标准 LR，公式自洽；
- **近似误差量化**：估计偏高比例 $= \frac{1}{1-p}$，pCTR = 5% 时仅高估 5.3%，pCTR = 50% 时高估一倍——与 6.1 节的数值实验（6.28 vs 12.61）精确吻合。

这个推导同时给出了 WLR 的一个隐藏前提：模型用**单个标量** $e^{w\cdot x}$ 同时编码 $p(x)$ 与 $E[T \mid 1, x]$ 的乘积，要求两者的联合变化模式能被同一个线性打分表达——如果某特征提点击但降时长（两者方向相反），单标量表达力不足时会产生系统偏差。多头的显式建模方案（如 ZILN 的 $\pi$ 头与 $\mu$ 头各管各的）天然免疫这个问题。


---

## 五、Serving 方式

### 5.1 为什么用 $e^{Wx+b}$ 而非 sigmoid

| 方式 | 公式 | 计算的是 |
|------|------|---------|
| 标准 LR Serving | $\text{sigmoid}(Wx+b) = \frac{1}{1+e^{-(Wx+b)}}$ | 概率 $p$ |
| YouTube Serving | $e^{Wx+b}$ | **Odds** ≈ 期望观看时长 |

由于 Weighted LR 训练后，$e^{Wx+b}$ 等于加权后的 Odds，而加权后的 Odds 近似等于期望观看时长，因此 Serving 时直接使用指数形式即可。

### 5.2 排序而非精确值

Serving 时只需要相对排序（哪个视频期望观看时长更长），不需要绝对值，所以 $e^{Wx+b}$ 直接当排序分用就行。


---

## 六、前提条件与局限性

### 6.1 核心前提：pCTR 必须很小

Weighted LR 的 $Odds \approx E(T_i)$ 近似仅在 pCTR 很小时成立。当 pCTR 较大时（如 50%），近似严重失效。

**数值实验验证**（来自 srome.github.io）：

| pCTR | 估计观看时长 | $e^{\text{bias}}$ | 是否吻合 |
|------|------------|-------------------|---------|
| 0.1%（小） | 0.0582 | 0.0582 | 吻合 |
| 50%（大） | 6.28 | 12.61 | 严重偏离 |

### 6.2 为什么不直接回归预测时长（MSE loss 的问题）

直接用 MSE loss 回归预测播放时长存在两个问题：

1. **分布假设**：MSE 假设预估 label 和误差项符合**正态分布**，但播放时长通常服从**长尾分布**（如几何分布），分布假设不匹配
2. **离群值敏感**：MSE 对离群值（超长观看时长）非常敏感，容易被极端值拉偏
3. 分类问题比回归问题更易求解，预测准确率更高

Weighted LR 把时长预测转成了分类问题：分类框架现成、求解稳定，时长信息通过权重进模型，两头都占住了。


### 6.3 WCE 损失函数推导

YouTube 论文提出的 WCE（Weighted Cross Entropy）损失函数，将正样本的 label 置为 $t_i$（观看时长），负样本 label 为 0：

$$C = -\sum_{i=1}^{n} \left( t_i \cdot y_i \cdot \log f(x_i) + (1 - y_i) \cdot \log(1 - f(x_i)) \right)$$

其中 $f(x) = \frac{1}{1+e^{-\theta x}}$，推导 Odds：

$$\frac{f(x)}{1 - f(x)} = e^{wx} = \frac{\sum_{i=1}^{k} t_i}{n - k} \approx E(t)$$

其中 $n$ 是总样本数，$k$ 是正样本数。得出可以用 $e^{wx}$ 来表示预估的观看时长。

**「odds = Σt/(n−k)」的严格来源**——这一步不是近似，而是加权似然对全局截距的一阶条件。考虑只有截距的模型（全局单一打分 $b$，$z = e^b$），加权损失为：

$$C(b) = -\sum_{i=1}^{n} \Big[ t_i y_i \ln \sigma(b) + (1 - y_i)\ln(1 - \sigma(b)) \Big]$$

对 $b$ 求导（利用 $\sigma'(b) = \sigma(1-\sigma)$，正负两项的求导会分别产生互补因子）：

$$\frac{\partial C}{\partial b} = -\sigma(1-\sigma)\sum_i \frac{t_i y_i}{\sigma} + \sigma(1-\sigma)\sum_i \frac{1-y_i}{1-\sigma} = 0$$

整理（两边同除 $\sigma(1-\sigma)$）：

$$\frac{\sum_i t_i y_i}{\sigma(b)} = \frac{n-k}{1-\sigma(b)} \implies \sigma(b) = \frac{\sum_i t_i y_i}{\sum_i t_i y_i + (n-k)}$$

于是全局 odds：

$$e^b = \frac{\sigma(b)}{1-\sigma(b)} = \frac{\sum_{i=1}^{k} t_i}{n-k}$$

模型收敛时，全局 odds 精确等于「加权正样本总量 ÷ 负样本数」，也就是「平均每次曝光对应的期望观看时长」的经验版本。


### 6.4 梯度不对称问题（WCE 的重要局限）

WCE loss 在**低估**和**高估**时梯度不对称，容易导致模型高估。

改进方法：为每一个正例增加一个负例，用 $y$ 表示时长，loss 变为：

$$C = -\sum_{i=1}^{n} \left( y_i \cdot \log f(x_i) + \log(1 - f(x_i)) \right)$$

令 $y'_i = e^{\theta x_i}$，对 $y'_i$ 求梯度：

$$\frac{\partial C}{\partial y'_i} = \frac{y'_i - y_i}{y'_i \cdot (1 + y'_i)}$$

**推导过程**：由 $f = \sigma(\theta x) = \frac{y'}{1+y'}$ 得 $\ln f = \ln y' - \ln(1+y')$、$\ln(1-f) = -\ln(1+y')$，代入损失：

$$C = -\sum_i \Big[ y_i \ln y' - (1 + y_i)\ln(1+y') \Big]$$

对 $y'$ 求导（两项分别用 $\frac{d}{du}\ln u = \frac{1}{u}$）：

$$\frac{\partial C}{\partial y'} = -\sum_i \Big[ \frac{y_i}{y'} - \frac{1+y_i}{1+y'} \Big] = -\sum_i \frac{y_i(1+y') - (1+y_i)y'}{y'(1+y')} = \sum_i \frac{y' - y_i}{y'(1+y')}$$

| 情形 | 梯度大小 | 影响 |
|------|---------|------|
| 低估（$y' < y$） | 梯度很大（分子为负且 $y'$ 小、分母小） | 快速修正低估 |
| 高估（$y' > y$） | 梯度很小（$y'$ 大、分母大，梯度被 $y'^2$ 压缩） | 高估难以修正 |

> 结论：低估时梯度大、高估时梯度小，模型倾向于把观看时长往高了估。用 WCE 之前先想清楚一件事：业务能不能容忍系统性高估。


### 6.5 几何分布假设

WCE 实际上假设了样本分布服从**几何分布**（Geometric Distribution）。如果实际样本分布不是几何分布（例如观看时长服从其他长尾分布），可能导致效果不好。

**为什么会有这个隐含假设**：由 4.4 节的推导，WLR 用**单个标量** $e^{w\cdot x}$ 同时表达「是否观看」$p(x)$ 与「观看多久」$E[T \mid 1, x]$ 的乘积。要让这个单标量结构自洽，时长分布族必须是**单参数**的——几何分布（「每一秒以固定概率继续观看」）是最简选择，其期望由单一参数决定，odds 恰好能写成该参数的单调函数。而多参数分布（如对数正态需要 $\mu$、$\sigma$ 两个参数）中，「是否观看」与「看多久」可以独立变化，单标量 odds 无法同时追踪两者——这正是 ZILN 之类的多塔结构（$\pi$ 头管是否观看、$\mu/\sigma$ 头管时长）存在的理由。

> 用之前先看一眼时长分布跟几何分布差多远，差得多就换建模方式，别硬套。


---

## 七、训练方法

Weighted LR 有两种训练方式：

| 方法 | 说明 | 优劣 |
|------|------|------|
| **重复采样** | 正样本按 weight 重复 sampling 后训练 | 简单，但增加样本量和训练时间 |
| **梯度加权** | 在梯度下降中改变梯度的 weight | 更推荐：减少样本处理量和梯度更新次数 |


---

## 八、推广：当 pCTR 不可忽略时

当业务场景中 pCTR 达到 20% 甚至更高时，不能忽略 $N_{pos}$ 的近似。

### 8.1 问题分析

原始 Odds 近似为：

$$\frac{\sum_{i=1}^{N_{pos}} w_i}{N_{neg}} \approx \frac{\sum_{i=1}^{N_{pos}} w_i}{N_{neg} + N_{pos}}$$

当 $N_{pos}$ 相对 $N_{neg}$ 不可忽略时，分母需要补上 $N_{pos}$。

### 8.2 解决方案

在负样本中**随机采样 $N_{pos}$ 个样本**填充分母，修正近似误差。

### 8.3 采样偏差修正

当训练数据经过负采样（下采样）后，需要通过 **inverse propensity scoring** 修正权重：

- 正样本权重：$w_i \times \frac{p_1}{\hat{p}}$（原始 pCTR / 采样后 pCTR）
- 负样本权重：$1 \times \frac{p_2}{\hat{p}}$

数值实验验证：修正后估计观看时长 0.602 vs $e^{\text{bias}}$ 0.609，吻合良好。

### 8.4 为每个正例增加负例（解决近似误差）

另一种改进方法：为每一个正例增加一个负例来解决 $\frac{\sum t_i}{n-k} \approx E(t)$ 的近似问题。

改进后的损失函数：

$$C = -\sum_{i=1}^{n} \left( y_i \cdot \log f(x_i) + \log(1 - f(x_i)) \right)$$

其中 $y_i$ 直接表示时长。这样每个正样本都配有对应的负样本，使得正负比例更均衡，近似更准确。

> 但这个方法有梯度不对称问题（见 6.4 节）：低估修得快、高估修得慢，高估会越积越多。


---

## 九、时长建模的其他方案与 Weighted LR 的演进

Weighted LR 是 YouTube 论文提出的经典方法，但工业界在此基础上发展出了多种改进和替代方案。

### 9.1 Weighted LR 的推广：GMV 场景

Weighted LR 的思路不限于观看时长，可推广到其他业务目标：

- **推荐商品时**，对"点击正样本"加权，权重设为该商品的**历史成交金额（GMV）**
- 有利于将用户喜欢且高价值的商品排序在前，增加商家利润


### 9.2 CTR 不可忽略时的具体实现

当 pCTR 不可忽略时，线上预估需要修正：

$$\text{预估时长} = \text{odds} \times (1 - \text{ctr})$$

具体实现要点：
- label 是 ctr label，但时长加权，时长取**根号**压缩
- 线上预估时：$(1 - \text{ctr}_q) \times e^{\text{dura-tower 输出}}$


### 9.3 回归建模方式（log2 + MSE）

除了将时长预估转为分类问题，还可以直接回归建模：

1. 对原始观看时长取 **log2** 作为 label（压缩波动范围）
2. 使用 **MSE loss** 对时长目标进行回归学习
3. 时长限制在 1~120 秒，取 log2 后范围在 0~7.9 之间
4. 为什么不用 log10：取值范围仅 (0, 2.1)，范围太小

> 优势：对原始时长取 log2 压缩后，时长波动范围缩小，模型预估更稳定。


### 9.4 完播率建模

另一种时长相关目标：完播率（用户是否看完视频）。

| 方法 | 说明 | 特点 |
|------|------|------|
| 回归方法 | 预估完播率连续值 | 通用性强 |
| 二元分类 | 是否完播（0/1） | 有利于短视频，不利于长视频 |

> 短视频天然完播率高，长视频完播率低，二元分类会偏向短视频。


### 9.5 时长归一化（Duration Normalization）

**问题**：长视频的消费时长天然更高，导致 Weighted Logloss 中长视频比重过大，偏向优化长视频。

**方案**：
1. 按 **video duration 分桶**（如 0-1 分钟、1-5 分钟等）
2. 每桶内部按消费时长分段
3. 将消费时长**归一化到 0~1** 范围
4. 用归一化后的值作为 label 训练

**特点**：
- 消除视频本身长短带来的偏差
- 可以叠加在 Weighted Logloss、CREAD、EMD 等其他方法上
- 线上预估时需要逆操作还原真实时长

> Weighted Logloss 在归一化场景下存在理论缺陷：$e^{wx}$ 不能保证预估值在 [0,1] 范围内，需人工变换（如 sigmoid）。


### 9.6 CREAD（快手 AAAI'24）

CREAD 将时长建模转化为**多个有序二分类任务**：

1. 将消费时长从小到大排序，通过 $m+1$ 个阈值分成 $m+1$ 个区间
2. 每个区间构造一个二分类任务：$y_j = \mathbb{1}(t > \text{threshold}_j)$，共 $m$ 个
3. 线上预估时通过累积概率还原时长

**额外损失**：
- **Huber loss**：还原时长与真实时长的偏差
- **Hinge loss**：保证二分类之间的序关系


### 9.7 EMD（Earth Mover's Distance）

EMD 从**类别间有序关系**角度改进多分类：

- 传统多分类中类别间无关系，但时长区间有自然顺序
- 使用 **Squared EMD Loss**：真实区间附近概率应更高，向两侧平滑降低
- 与 CREAD 类似的分桶方式，但无需额外 Huber/Hinge loss

**EMD 相比 CREAD 的优势**：
1. 不需要额外的 Huber loss 和 Hinge loss
2. 可与 **Distill Softmax** 合用，效果更佳


### 9.8 各方案对比总结

| 方案 | 核心思路 | 优势 | 劣势 |
|------|---------|------|------|
| **Weighted LR** | 时长作为正样本权重，Odds ≈ E(T) | 简单高效，分类转回归 | pCTR 须很小；梯度不对称；假设几何分布 |
| **log2 + MSE 回归** | log2 压缩时长 + MSE 回归 | 预估稳定 | 回归问题更难求解 |
| **完播率建模** | 预估完播概率 | 简单 | 偏向短视频 |
| **时长归一化** | 分桶归一化消除长视频偏差 | 消除 duration bias | 需逆操作还原 |
| **CREAD** | 多个有序二分类 + Huber + Hinge | 精细建模 | 实现复杂 |
| **EMD** | 有序多分类 + EMD Loss | 无需额外损失；可叠加 Distill Softmax | 分桶设计敏感 |

### 9.9 后续研究论文

| 论文 | 会议 | 来源 | 方向 |
|------|------|------|------|
| Deconfounding Duration Bias in Watch-time Prediction | KDD'22 | [ar5iv](https://ar5iv.labs.arxiv.org/html/2206.06003) | 因果消偏 |
| CREAD | AAAI'24 | 快手 | 多有序二分类 |
| Generative Regression Based Watch Time Prediction | WWW'25 | [arXiv](https://arxiv.org/html/2412.20211v1) | 生成式回归 |
| Relative Advantage Debiasing | — | [arXiv](https://arxiv.org/html/2508.11086v2) | 相对优势去偏 |

---

## 十、完整逻辑链总结

```
伯努利分布 p
    │
    ▼
Odds = p/(1-p) ──ln──→ logit = ln(Odds) = θᵀx
    │                              │
    ▼                              ▼
sigmoid(θᵀx) = p              e^(θᵀx) = Odds ← YouTube Serving 函数
                                   │
    Weighted LR: wᵢ = Tᵢ (正样本)     │
    │                              ▼
    ▼                         Odds ≈ wᵢ·p = Tᵢ·p = E(Tᵢ)
Odds(i) = wᵢp/(1-wᵢp)             │
    │                              ▼
    ▼                         期望观看时长 → 排序推荐
p 很小时: ≈ wᵢ·p
```

总结成一句：YouTube 用观看时长作为正样本权重训练 LR，模型输出的 Odds 近似等于期望观看时长，Serving 时直接用 $e^{Wx+b}$（即 Odds）排序。时长预测就这么转成了分类问题。

---

## 十一、来源索引

正文各章节的依据来源汇总如下（正文不再逐节重复列出）：

| # | 来源 | 类型 | 链接 | 主要引用章节 |
|---|------|------|------|--------------|
| 1 | 揭开YouTube深度推荐系统模型Serving之谜 | 知乎专栏（王喆） | https://zhuanlan.zhihu.com/p/61827629 | 一、二、三、五、七 |
| 2 | Predicting Watch Time like YouTube via Weighted Logistic Regression | 技术博客（srome） | http://srome.github.io/Predicting-Watch-Time-Like-YouTube-With-Weighted-Logistic-Regression/ | 3.1、四、五、6.1、6.2、八 |
| 3 | weighted—LR的理解与推广 | 博客园（Jamest） | https://www.cnblogs.com/hellojamest/p/11871108.html | 一、二、三、6.2、七、八 |
| 4 | Deep Neural Networks for YouTube Recommendations | 原始论文（Covington et al., 2016） | YouTube 2016 RecSys | 全文背景 |
| 5 | Weighted LR（WCE Weighted cross entropy） | 博客园（AI_Engineer） | https://www.cnblogs.com/xumaomao/p/15207305.html | 6.2、6.3、6.4、6.5、八 |
| 6 | 时长建模总结 | 知乎 | https://zhuanlan.zhihu.com/p/1903020329921650788 | 9.1、9.2、9.3、9.4 |
| 7 | 推荐系统时长建模的常用方案 | mathmach.com | https://mathmach.com/f5a7a06e/ | 9.5、9.6、9.7、9.8 |
| 8 | Deconfounding Duration Bias in Watch-time Prediction | KDD'22 论文 | https://ar5iv.labs.arxiv.org/html/2206.06003 | 9.9 |
| 9 | Generative Regression Based Watch Time Prediction | WWW'25 论文 | https://arxiv.org/html/2412.20211v1 | 9.9 |
