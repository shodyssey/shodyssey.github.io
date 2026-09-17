---
title: 从 NLP 到 LLM 系列（四）：Transformer 与自注意力机制
date: 2026-09-16 15:00:00
categories:
  - LLM 系列
tags:
  - Transformer
  - 自注意力
  - 缩放点积
  - 位置编码
  - ConvS2S
description: '从 NLP 到 LLM 系列第 4 篇——QKV 三角色设计与自注意力的逐步走查、√d_k 缩放的方差与雅可比推导（原论文 Footnote 4 与 suspect 措辞）、多头注意力的子空间动机、sinusoidal 位置编码、残差与 LayerNorm 的架构组装、ConvS2S 历史擂台对照、O(n²) 代价与三个死穴'
---

# 从 NLP 到 LLM 系列（四）：Transformer 与自注意力机制

> 从 NLP 到 LLM 系列 · 第 4 篇
> 前情提要：第 3 篇的 RNN 家族留下三个死穴——串行无法并行、长程依赖只是缓解、注意力在 RNN 里只是补丁。三个死穴指向同一个追问：如果注意力本身就是主角，会怎样？
> 下篇预告：GPT/BERT 与预训练范式。

---

## 一、2017 年 6 月的答案

第 3 篇结尾的追问，2017 年 6 月有了一篇论文标题式的回答：Attention is All You Need [1]。

答案的形态比标题更彻底：不仅注意力是主角，递归和卷积这两个"顺序结构"被整体扔掉。剩下的组件——检索（注意力）、加工（前馈网络）、稳定器（残差与归一化）——恰好全是可并行的矩阵运算。GPU 等了二十年的架构，就这么来了。

先看全貌，再拆零件：

![Transformer 总体结构](/img/series-04/the_transformer_3.png)

*图 1：Transformer 总体结构——编码器栈 + 解码器栈（图源：jalammar.github.io，CC BY-NC-SA）*

这一篇按"机制拆解 → 推导 → 架构组装 → 历史真相 → 代价"的顺序展开。先讲清自注意力这一件事，其余全部是围绕它的工程配套。

## 二、自注意力：序列内部的全连接检索

### 2.1 核心问题：不用递归，怎么让每个词看到所有词

RNN 的方案是"传话"：信息沿着时间步一站一站传。第 3 篇已经算过这笔账——传话路径 O(n)，梯度连乘衰减，传不远。

自注意力的方案是"开会"：每个词直接和句子里所有词交互一次，一步到位。路径长度 O(1)，没有中间站，也就没有连乘。

代价是计算量：n 个词两两交互，O(n²) 起步。这个代价值不值，第七章用原论文的复杂度对照表算，先记住这笔账。

### 2.2 QKV：一次检索的三种角色

每个词要"发出查询"又要"应答别人的查询"，还得携带"被检索的内容"。一次检索拆成三个角色：

- **Query（查询）**：当前词发出的"我想找什么"；
- **Key（索引）**：每个词对外展示的"我是什么类型"；
- **Value（内容）**：每个词真正携带的信息。

我的看法是，QKV 这个三分法是本篇最值得慢看的设计。它把第 3 篇 Bahdanau 的注意力正式化了：那里解码器状态 s_{t-1} 干的就是 Query 的活，编码器隐藏状态 h_j 同时充当 Key 和 Value——一个向量身兼两职。Transformer 把这两职拆开（Key 管匹配、Value 管内容），匹配能力和内容表达就解耦了，多头注意力（第四章）才有施展空间。

顺带回答一个常见疑问：为什么 Q、K 要用两个不同的投影矩阵，直接让每个词查自己不行吗？不行，原因很硬：如果 Q = K（同一矩阵投影），每个词和自己的点积 q_i·k_i = ||q_i||² 永远是行内最大——**每个词最匹配的都是它自己**，注意力矩阵对角线独大，跨词的信息流动被自己堵死。Q、K 分开投影，等于给"提问"和"应答"两套不同的表示，模型才有机会学到"我该关注别人身上的什么"。Value 再单独一套，是因为"拿什么内容去融合"和"怎么被匹配上"也是两件事——一个词可能 Key 很显眼（高频虚词），但 Value 没什么信息量，分开后注意力不会被虚词的内容污染。

与 Bahdanau 的另一处本质区别：Bahdanau 是 cross-attention（解码器查编码器，跨序列），自注意力是 self-attention（序列内部自己查自己，同序列）。后者才是"扔掉递归"的关键——序列不再需要外部记忆，检索对象就是自己。

自注意力学出来的"关注"长什么样？原论文的可视化给过一个经典例子：句子 "The animal didn't cross the street because **it** was too tired"，看 "it" 这个词的自注意力分布——权重最高的落点是 "animal"。指代消解这种传统 NLP 里要专门建模的任务，在自注意力里作为副产品出现了。第 3 篇 Bahdanau 的对齐热力图展示过"翻译对齐"的涌现，这里是"指代关系"的涌现——**注意力机制学到的软性结构，总是超出训练目标的字面要求**。

### 2.3 逐步走查：一句话里的自注意力

用一个具体例子把流程走一遍。句子 "Thinking Machines"，看 "Thinking" 这个词怎么获得融合了上下文的新表示。

**第一步：每个词生成三个向量。** 词向量分别乘三个投影矩阵，得到 q、k、v：

![自注意力向量](/img/series-04/transformer_self_attention_vectors.png)

*图 2：每个词经三个投影矩阵生成 Query、Key、Value（图源：jalammar.github.io，CC BY-NC-SA）*

**第二步：算匹配分。** "Thinking" 的 q₁ 与每个词的 k 做点积——点积大，匹配强：

![匹配打分](/img/series-04/transformer_self_attention_score.png)

*图 3：q₁ 与全部 k 点积，得到匹配分（图源：jalammar.github.io，CC BY-NC-SA）*

**第三步：除以 √d_k，softmax 归一。** 匹配分变成权重（为什么除以 √d_k 是第三章的全部内容）：

![softmax 归一](/img/series-04/self-attention_softmax.png)

*图 4：打分除以 8（d_k=64）后 softmax，得到注意力权重（图源：jalammar.github.io，CC BY-NC-SA）*

**第四步：加权求和 Value。** 权重乘各词的 v，加起来，就是 "Thinking" 的新表示 z₁：

![加权求和](/img/series-04/self-attention-output.png)

*图 5：z₁ = Σ α·v，融合了全句信息的新表示（图源：jalammar.github.io，CC BY-NC-SA）*

矩阵形式一次算完全句（X 是 n×d 的词向量矩阵）：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

![矩阵计算](/img/series-04/self-attention-matrix-calculation.png)

*图 6：矩阵形式的完整计算链（图源：jalammar.github.io，CC BY-NC-SA）*

![矩阵计算二](/img/series-04/self-attention-matrix-calculation-2.png)

*图 7：一路矩阵乘到底（图源：jalammar.github.io，CC BY-NC-SA）*

四步走完，一句话的每个词都拿到了看过全句的新表示。没有递归，没有传话，纯矩阵乘法——GPU 最擅长的事。

## 三、为什么除以 √d_k

原论文对这个模块的官方画法——注意中间那个 Scale 步骤，就是本章要解释的东西：

<img src="/img/series-04/paper-fig2-scaled-dot-product.png" alt="缩放点积注意力官方结构" style="max-width: 320px; display: block; margin: 0 auto;">

*图 8：Scaled Dot-Product Attention：MatMul → Scale → Mask → SoftMax → MatMul（图源：原论文 Figure 2 左 [1]）*

这一章是本篇的推导核心。结论先放这：不除以 √d_k，点积的方差随维度增长，softmax 会饱和成 one-hot，梯度跟着消失。

### 3.1 原论文自己是怎么说的

原论文 3.2.1 节的原文（已核对）：

> We suspect that for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients.

注意措辞：**suspect，"我们猜想"**。连作者都没给严格证明。方差分析的完整表述也不在正文，在 **Footnote 4**（脚注）里：设 q、k 的各分量独立、均值 0、方差 1，则点积 q·k 的均值为 0、方差为 d_k。

还有一个容易被略过的前文：原论文先引了 Bahdanau 的 additive attention，说"大 d_k 时 additive 优于不缩放的点积 [3]"，然后才给出 suspect 和缩放。逻辑链是：additive 的优势现象在前，猜想在后，缩放是对策。

### 3.2 方差分析：点积为什么会变大

设 q、k 的各分量是独立随机变量，均值 0、方差 1。点积是 d_k 个独立乘积之和：

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

逐项算（完整推导见附录 A）：每个乘积项 q_i·k_i 的期望为 0（独立且均值 0 的乘积期望 = 期望之积 = 0）；方差为 1（E[q_i²k_i²] = E[q_i²]E[k_i²] = 1，再减去均值平方 0）。独立项求和，方差相加：

$$E[q \cdot k] = 0, \qquad \text{Var}(q \cdot k) = d_k$$

标准差就是 √d_k。d_k = 64 时点积的典型幅度已经是 ±8 量级，d_k = 512 时是 ±23——softmax 的输入跑到了这个量级，事情就坏了。

### 3.3 softmax 饱和：梯度怎么没的

softmax 的输出对输入的雅可比（附录 A 完整推导）：

$$\frac{\partial s_i}{\partial z_j} = s_i\big(\mathbb{1}\{i = j\} - s_j\big)$$

输入方差大意味着什么：某个 z 比其他大出几个标准差，softmax 输出就逼近 one-hot——最大的 s_i ≈ 1，其余 ≈ 0。代回雅可比：s_i(1−s_i) ≈ 0，−s_i·s_j ≈ 0，**整个雅可比矩阵趋零**。前一层传来的梯度，在 softmax 这里几乎全被吞掉。

除以 √d_k 之后：

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

方差拉回 1，与维度无关。那为什么不是 2√d_k，或者干脆除以 d_k？把两端都算一遍就清楚了：除以 2√d_k 把方差压到 1/4，softmax 输入被压得过平，所有权重趋向均匀的 1/n，注意力失去区分度——代回雅可比，均匀分布下对角项 (1/n)(1−1/n) ≈ 1/n，梯度被序列长度稀释，同样学不动。**缩放的两端是两种死法**：不缩放死于饱和（one-hot，雅可比归零），过度缩放死于均匀（1/n，雅可比被稀释成 1/n 量级）。√d_k 恰好落在中间——方差 1，softmax 既不饱和又有区分度，任何别的因子都会向其中一端漂移。

数值例子感受一下（d_k = 64）：方差 1 的 softmax 输出是平缓的概率分布，梯度健康；方差 64（不缩放）的输出已经接近 one-hot，雅可比几乎归零；方差 1/16（过度缩放）的输出接近均匀，注意力权重全部 ≈ 1/n，模型退化成"平均池化"，雅可比同样微弱。三种状态里，只有中间那种能训练。

### 3.4 一条脚注撑起的工程基石

回头看，缩放因子 √d_k 是整个 Transformer 里最不起眼的部分，但训练稳定性全靠它。据团队回忆（访谈转述，细节有文学化加工），这个因子是 Noam Shazeer 在项目最艰难的时候丢进来的——当时模型反复数值爆炸，他路过会议室问了一句"你们点积的方差是不是随维度线性增长"，然后建议除以 √d_k 试试。

我的理解是，这个故事里真正值得记的不是灵光一现，而是问题定位的方式：训练崩了，先查分布的量级，再查梯度路径——方差分析这种"土办法"能定位到的问题，不需要玄学解释。

## 四、多头注意力：一次检索不够就做八次

### 4.1 单头的问题：平均抹平了子空间

原论文的动机句（已核对）：

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. **With a single attention head, averaging inhibits this.**

翻译过来：单头注意力对所有位置做一次加权平均，平均操作抹平了"不同表示子空间"的差异。一个词和它的邻居，可能句法上相关（主谓）、语义上相关（同义）、指代上相关（代词回指）——这些关系活在不同的子空间里，一次检索只能抓住一种。

### 4.2 机制：八组低维投影并行

做法直接：把 d_model = 512 砍成 8 份，每份 64 维，做 8 组独立的注意力，拼回来：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q,\ K W_i^K,\ V W_i^V)$$

![多头 QKV](/img/series-04/transformer_attention_heads_qkv.png)

*图 9：每个头用独立的低维投影（图源：jalammar.github.io，CC BY-NC-SA）*

<img src="/img/series-04/paper-fig2-multi-head.png" alt="原论文多头结构" style="max-width: 380px; display: block; margin: 0 auto;">

*图 10：Multi-Head Attention 结构图（图源：原论文 Figure 2 右 [1]）*

为什么用低维投影而不是 8 个全维头：8 组 64 维的总计算量 ≈ 1 组 512 维，多头几乎免费。原论文的可视化（论文 Figure 3 附近）显示不同头确实学到了不同类型的依赖——两个头关注前一个词，另一些头关注句法相关的远端词。

![多头总结](/img/series-04/transformer_multi-headed_self-attention-recap.png)

*图 11：八头各自算完再拼接（图源：jalammar.github.io，CC BY-NC-SA）*

### 4.3 消融实验：头数不是越多越好

"多头有用"不是修辞，原论文 Table 3 用一组消融把它钉死了（英德翻译 dev 集，固定总计算量，d_k = d_model/h，已核对）：

| 配置 | d_k | dev PPL | dev BLEU |
|------|-----|---------|----------|
| 1 头 | 512 | 5.29 | 24.9 |
| 4 头 | 128 | 5.00 | 25.5 |
| **8 头（base）** | **64** | **4.92** | **25.8** |
| 16 头 | 32 | 4.91 | 25.8 |
| 32 头 | 16 | 5.01 | 25.4 |

两个方向的读法：**单头比最佳配置差 0.9 BLEU**——平均抹平子空间的代价是实打实的；但 32 头也开始掉——每个头的维度被压到 16，单头的匹配能力先崩了。原论文自己的结论也是这句：单头明显差，头太多也伤。8~16 头是甜点区，这个结论后来在各类变体里反复被验证。

同一张消融表里还有一行值得记：把 d_k 固定砍半（不管头数），PPL 从 4.92 涨到 5.16——原论文的解读是"判断兼容性这件事本身不容易"，**Key 的维度预算不能省**。多头省的是"视角数"，不是"每个视角的分辨率"。

## 五、位置编码：没有递归，顺序从哪来

自注意力有个先天缺陷：**置换不变**。打乱词序，注意力的输出不变——"猫追狗"和"狗追猫"在它眼里一样。顺序信息必须从外部注入。

方案是在输入端给每个位置加一个向量：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \qquad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

不同维度用不同频率的 sin/cos——低维振荡快（区分相邻位置），高维振荡慢（编码长程位置）：

![位置编码示例](/img/series-04/transformer_positional_encoding_large_example.png)

*图 12：位置编码矩阵——每行一个位置，每列一个频率（图源：jalammar.github.io，CC BY-NC-SA）*

![原论文位置编码](/img/series-04/attention-is-all-you-need-positional-encoding.png)

*图 13：不同维度频率下的波形（图源：原论文 Figure 5 局部 [1]）*

选三角函数不是审美偏好，是功能需求：**PE(pos+k) 可以表示成 PE(pos) 的线性函数**（推导见附录 B）。这意味着模型要学"相对位置"关系时，只需要学一个固定的线性变换，而这个变换对任意偏移 k 通用。学到的相对模式能平移复用，这是 learned embedding（每个位置一个独立参数）做不到的。

频率的设计也值得看一眼：d_model/2 对维度，每对的角频率从 1 等比递减到 1/10000（公比 10000^(−2/d_model)）。高角频率的维度在 pos=0~10 的短程内就完成一个周期，负责区分相邻位置；低角频率的维度要到 pos 上千才振荡一次，负责长程位置。**整个位置范围被一组频率铺满**——10000 这个底数不是调参调出来的，是"覆盖足够长序列"的工程余量，苏剑林对这组公式的追根溯源讲得最透 [5]。

原论文 Table 3 也把 learned 和 sinusoidal 正面比过（已核对）：learned positional embedding 的 dev PPL 4.92、BLEU 25.7，sinusoidal 的 4.92、25.8——**两个方案几乎打平**。选 sinusoidal 的理由是它可能对训练长度之外的序列有更好的外推，原论文原话是 "may allow the model to extrapolate"——又一个"可能"。这个"可能"后来被证明只对了一半，位置编码的演进（相对位置编码、RoPE）成为后续长上下文的主战场之一。本篇记住一句即可：**位置编码是自注意力架构上的一块补丁，补丁的形态一直在换，需求本身没变**。

## 六、架构组装：把零件拼成引擎

### 6.1 编码器-解码器的骨架

![编解码器](/img/series-04/The_transformer_encoders_decoders.png)

*图 14：编码器栈与解码器栈（图源：jalammar.github.io，CC BY-NC-SA）*

![堆叠结构](/img/series-04/The_transformer_encoder_decoder_stack.png)

*图 15：6 层编码器 + 6 层解码器（图源：jalammar.github.io，CC BY-NC-SA）*

原论文的配置：6 层编码器 + 6 层解码器，d_model = 512，8 头，FFN 中间层 2048。

编码器每层两个子模块：多头自注意力 + 前馈网络。解码器每层三个：**带 mask 的**自注意力、cross-attention（Query 来自解码器，Key/Value 来自编码器输出——这是 Bahdanau 那种跨序列注意力在纯注意力架构里的位置）、前馈网络。

![编码器](/img/series-04/Transformer_encoder.png)

*图 16：编码器单层结构（图源：jalammar.github.io，CC BY-NC-SA）*

![解码器](/img/series-04/Transformer_decoder.png)

*图 17：解码器单层结构——比编码器多一个 cross-attention（图源：jalammar.github.io，CC BY-NC-SA）*

### 6.2 mask：训练时怎么既并行又因果

第 3 篇审查时留过一个坑：训练时解码器怎么并行？现在填上。

解码器的自注意力必须因果——生成第 t 个词时只能看前 t−1 个词。但训练时为了并行，整句一起前向。冲突的解法是 **mask 矩阵**：softmax 之前，把注意力分数矩阵的上三角（未来位置）全部置为 −∞：

$$\text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right), \qquad M_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}$$

−∞ 过 softmax 后权重为 0，未来位置对当前词的贡献被精确清零。训练时整句并行计算，因果性由 mask 保证——**并行和因果不是靠串行换来的，是靠一个上三角矩阵换来的**。这一招后来成为所有 GPT 系模型的标配。

### 6.3 残差与 LayerNorm：深度堆叠的稳定器

每个子模块的输出套一层结构：

$$\text{output} = \text{LayerNorm}\big(x + \text{Sublayer}(x)\big)$$

![残差与归一化](/img/series-04/transformer_resideual_layer_norm.png)

*图 18：每个子模块外面包一层"加残差 + 归一化"（图源：jalammar.github.io，CC BY-NC-SA）*

残差连接是第 3 篇 LSTM 传送带思想的直系后代：恒等通路让梯度走加法通道，不衰减。区别是 LSTM 用可学习的门控（f_t）控制通过量，残差干脆全通过（系数恒为 1）。LayerNorm 把每层的输出分布拉回稳定区间，防止 6 层 × 2~3 个子模块的堆叠把数值越滚越偏。

一个值得知道的版本细节：2017 原论文用的是 **Post-Norm**（归一化在残差相加之后，即上式）。后来的工作发现把归一化挪到残差相加之前（Pre-Norm）能让几十层的堆叠更稳定，GPT 系列用的就是 Pre-Norm。本篇不展开，只提醒一句：网上流传的"Transformer 诞生故事"里有人把 Pre-Norm 说成 2017 年的原始设计，这是错的。

### 6.4 FFN：注意力负责路由，FFN 负责加工

每个位置独立过一个两层全连接（中间 ReLU）：

$$\text{FFN}(x) = \max(0,\ x W_1 + b_1)\, W_2 + b_2$$

它逐位置（position-wise）独立作用，不跨位置交流信息。分工很清楚：**注意力做跨位置的路由（谁的信息流向谁），FFN 做逐位置的加工（收到信息后怎么变换存储）**。参数的大头也在这里——FFN 的两层（512→2048→512）占了每层约 2/3 的参数量。

组装完成，全貌如下：

<img src="/img/series-04/paper-fig1-transformer-architecture.png" alt="原论文架构图" style="max-width: 500px; display: block; margin: 0 auto;">

*图 19：Transformer 完整架构（图源：原论文 Figure 1 [1]）*

### 6.5 训练配方：让这个架构真的训得动的细节

架构图之外，原论文第 5 节的训练配方里有三个细节，后来都成了行业的默认设置，值得单独记录（均已核对原文）：

**Adam 的 β₂ = 0.98，不是默认的 0.999**。这不是笔误：训练初期二阶矩的滑动平均还没建立，β₂ 越大估计越滞后，warm-up 阶段的步长会失真。把 β₂ 从 0.999 降到 0.98，让二阶矩跟上学习率的快速变化——这个细节后来被无数复现失败的团队重新发现。

**学习率调度是 warm-up + 平方根衰减**（Noam 调度）：

$$lrate = d_{model}^{-0.5} \cdot \min\left(step^{-0.5},\ step \cdot warmup^{-1.5}\right)$$

前 4000 步线性升温，之后按步数平方根倒数衰减。为什么必须 warm-up：初期参数随机、梯度混乱，大学习率直接把 Adam 的矩估计带偏；先小步热身，等二阶矩稳定了再加速。**Transformer 是个对学习率调度敏感的架构**，这一点与它"没有归纳偏置、一切靠学"的性格一致——没有先验兜底，优化动力学就得自己稳住自己。

**正则化三件套**：残差 dropout 0.1（每个子层输出、加残差前）、embedding dropout 0.1、label smoothing 0.1。消融数字（Table 3 D 行）：dropout 从 0.1 砍到 0，dev PPL 从 4.92 恶化到 5.77——这个架构在 10 万步的训练量下，没有正则就过拟合。label smoothing 0.1 让模型对预测"留三分不确定"，代价是 BLEU 略降但 PPL 变好，翻译任务里这笔账算得过来。

解码用 beam search（beam size 4，长度惩罚 α=0.6）。训练侧并行、推理侧逐词，beam search 在候选序列层面找回一部分搜索质量。

## 七、历史真相：ConvS2S——被遗忘的擂台对手

讲清机制之后，值得还原一下 2017 年的真实赛场。多数解读把 Transformer 写成"打败 RNN 的革命者"，这个叙事省略了一个关键事实：**2017 年 RNN 已经不是主要对手了，真正的擂台对手是 Facebook 的卷积翻译模型 ConvS2S** [8,9]。

时间线：2017 年 5 月，Facebook 发 ConvS2S，机器翻译 SOTA，还特别强调了训练时间短 [9]。一个月后，Google 发 Transformer，英德翻译超 ConvS2S 2 BLEU，训练时间更短 [1]。这不是一个"新架构横空出世"的故事，是两家公司在同一赛道上的正面交锋。

ConvS2S 的思路：既然 RNN 慢在串行，就用卷积做序列编码——卷积天然并行。层级堆叠扩大感受野：a 层、核宽 k 的卷积，每个输出依赖 1 + a(k−1) 个输入。算一笔具体的账：ConvS2S 用核宽 5、堆 15 层以上，一个位置的"视野"也就 70 来个词——而自注意力一步就是全句。门控用 GLU（gated linear unit，卷积输出的两半做门控相乘）；位置信息用 learned embedding；attention 也在（decoder 每层一个 multi-step attention，encoder 最后一层的输出做 Key/Value）。**两个方案都带着注意力，胜负手不在"有没有注意力"，在"要不要卷积/递归"**。

原论文 Table 1 把三条路线的账算得很清楚（已核对）：

| 层类型 | 每层复杂度 | 串行操作数 | 最大路径长度 |
|--------|-----------|-----------|--------------|
| Self-Attention | O(n²·d) | O(1) | O(1) |
| Recurrent | O(n·d²) | O(n) | O(n) |
| Convolutional | O(k·n·d²) | O(1) | O(log_k(n)) |
| Self-Attention (restricted) | O(r·n·d) | O(1) | O(n/r) |

三行对比读下来：递归串行、路径长；卷积并行了，但路径 O(log_k n)——远距离信息仍要逐层扩散；自注意力一步直达，代价是 O(n²)。第四行 restricted 是原论文自己给的缓解方案（每个词只看半径 r 的邻域），这是后来稀疏注意力家族的官方起点。

原论文 Table 2 把这场对决的账目记得很全（已核对，newstest2014 测试集）：

| 模型 | EN-DE BLEU | EN-FR BLEU | 训练成本 (FLOPs) |
|------|-----------|-----------|------------------|
| GNMT + RL（RNN 集成） | 24.6 | 39.9 | 2.23×10¹⁹ / 1.4×10²⁰ |
| ConvS2S | 25.16 | 40.46 | 9.6×10¹⁸ / 1.5×10²⁰ |
| ConvS2S Ensemble | 26.36 | 41.29 | 7.7×10¹⁹ / 1.2×10²¹ |
| **Transformer (base)** | **27.3** | 38.1 | **3.3×10¹⁸** / 3.3×10¹⁸ |
| Transformer (big) | 28.4 | **41.8** | 2.3×10¹⁹ / — |

三个读法：**英德上 base 模型超 ConvS2S 2.14 BLEU，训练成本不到它的 1/3**（3.3×10¹⁸ vs 9.6×10¹⁸）；英法上 base 的 38.1 确实输给 ConvS2S 的 40.46——但注意成本，英法上 ConvS2S 花了 1.5×10²⁰ FLOPs，Transformer base 还是 3.3×10¹⁸，**45 倍的算力差距下只差 2.4 BLEU**；big 模型（d_model=1024、16 头、300K 步）在英法反超到 41.8，压过 ConvS2S 的集成。所以 2017 年的准确结论是：**Transformer 用零头的算力打平或超过了卷积方案**——效率优势才是它的第一张牌，精度优势要到预训练时代才彻底拉开。

实验结果里有个容易被略过的细节，照写：base 模型英法输给 ConvS2S。Transformer 不是全胜，2017 年的它只是"更省的那个新方案"。它的全面统治要到 BERT 和 GPT 时代才真正开始——那是第 5 篇的故事。

我的读法是，这场对决里最有信息量的不是谁赢，是两个团队对同一个问题（RNN 慢）给出的两条路线：Facebook 选择改良感受野（卷积层级扩散），Google 选择直接拆掉路径（全局直连）。一年后分出胜负的方式也不是精度——是"谁的架构更容易吃下数据和算力"。这跟第 2 篇 Word2Vec 的故事是同一个模式：**架构之争的最后裁决者，从来是可扩展性**。

## 八、代价与死穴

Transformer 解决了第 3 篇的三个死穴，也埋下了自己的三个。

**死穴一：O(n²) 复杂度**。算笔具体的账：n=512（2017 年翻译任务的典型长度）时，注意力矩阵是 512×512 ≈ 26 万个匹配分；n=128K（今天长上下文的量级）时是 128K×128K ≈ 160 亿个匹配分——长度涨 256 倍，注意力开销涨 6.5 万倍。序列长度翻倍，注意力计算量翻四倍。这是"全局直连"路线的固有代价——第 1 篇 n-gram 的窗口问题在这里换了形态重现：n-gram 是"想看远但参数爆炸"，自注意力是"能看远但计算爆炸"。原论文的 restricted attention（O(r·n·d)）是第一个官方缓解，后来的稀疏注意力、线性注意力都在这条线上。**结构解决了记忆问题，把成本转嫁给了计算**。

**死穴二：位置编码是补丁**。sinusoidal 的外推能力有限（原论文自己也只说了"可能"），训练长度之外的序列表现存疑。这个补丁后来被相对位置编码、RoPE 一路迭代——补丁形态在换，"顺序信息必须外部注入"这个需求本身从没消失。

**死穴三：数据与算力饥饿**。原论文的训练配置放在 2017 年就是重装备：base 模型 8 块 P100 跑 12 小时，big 模型同样 8 块卡跑三天半。更大的问题是这个架构没有归纳偏置（inductive bias），一切关系都要从数据里学，吃的数据越多越好。这个"饥饿"直接催生了下一场革命：既然架构这么能吃，那就喂它整个互联网——预训练范式登场。

还有一条常被忽略的：**推理时的解码器仍然是串行的**。训练并行了，但生成时 decoder 还是逐词自回归——每生成一个词都要重算一遍注意力，生成 n 个词的总开销是 O(n²)。

这个缺口催生了 KV cache：既然每个位置的 K、V 只依赖它自己和之前的词，跟还没生成的词无关，那就算过一次存下来。生成第 t 个词时，只算新词自己的 q、k、v，把 k、v 追加进缓存，q 只跟整个缓存做一次注意力——单步开销从 O(t²·d) 降到 O(t·d)，n 个词总共省掉一个平方因子。代价是显存：缓存大小随序列长度线性涨（层数 × 头数 × 序列长度 × d_k），长上下文时代 KV cache 的显存开销能超过模型权重本身，GQA、MLA 这些后来者的出发点之一就是压这份缓存。训练侧的并行红利没有自动传导到推理侧，这个缺口由 KV cache、投机解码等一系列工程手段填补——架构定了上限，工程把上限一点点兑现。

## 九、结语

收个总账。第 3 篇的三个死穴，本篇逐一解决：

| 第 3 篇的死穴 | 本篇的解法 |
|--------------|-----------|
| 串行计算无法并行 | 无递归纯矩阵运算，训练完全并行（mask 保因果） |
| 长程依赖只是缓解 | 任意位置直连，路径 O(1)，梯度不再连乘 |
| 注意力是 RNN 的补丁 | 注意力成为唯一主角，递归被整体扔掉 |

但 2017 年的 Transformer 还只是一个翻译模型的骨架。让它从"翻译专用架构"变成"通用序列引擎"的，不是架构本身的又一次革命，是训练方式的革命——用海量无标注文本做预训练。编码器拿去双向预训练成了 BERT，解码器拿去因果预训练成了 GPT，两条路在第 6 篇之后分别展开。

架构的故事到这里告一段落。下一篇：GPT/BERT 与预训练范式。

---

## 附录 A √d_k 缩放的完整推导

### A.1 点积的均值与方差

**假设**：q、k ∈ ℝ^{d_k}，各分量独立，均值 0、方差 1。

均值（期望线性 + 独立性）：

$$E[q \cdot k] = E\left[\sum_{i=1}^{d_k} q_i k_i\right] = \sum_{i=1}^{d_k} E[q_i k_i] = \sum_{i=1}^{d_k} E[q_i] E[k_i] = 0$$

方差（独立项方差相加）：

$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} \Big[ E[q_i^2]E[k_i^2] - \big(E[q_i]E[k_i]\big)^2 \Big] = \sum_{i=1}^{d_k} (1 \times 1 - 0) = d_k$$

其中 E[q_i²] = Var(q_i) + (E[q_i])² = 1。缩放后：

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{1}{d_k} \cdot d_k = 1$$

### A.2 softmax 的雅可比与梯度消失

记 z_i 为 softmax 的输入，s_i = exp(z_i) / Σ_l exp(z_l) 为输出。对 z_j 求偏导，分两种情况：

i = j 时（商法则，分母记 Z = Σ exp(z_l)）：

$$\frac{\partial s_i}{\partial z_i} = \frac{e^{z_i} Z - e^{z_i} e^{z_i}}{Z^2} = s_i(1 - s_i)$$

i ≠ j 时：

$$\frac{\partial s_i}{\partial z_j} = \frac{0 - e^{z_i} e^{z_j}}{Z^2} = -s_i s_j$$

合并：

$$\frac{\partial s_i}{\partial z_j} = s_i\big(\mathbb{1}\{i=j\} - s_j\big)$$

**饱和分析**：输入方差大 → 某个 z 值远超其他 → s_max → 1、其余 → 0。此时雅可比的每一项：对角项 s_i(1−s_i) → 0（s_i 要么趋 1 要么趋 0），非对角项 −s_i·s_j → 0。整个雅可比趋零，反向传播的梯度在 softmax 处断流。

缩放把输入方差固定为 1，与 d_k 无关——softmax 永远工作在"既不饱和又有区分度"的区间。原论文正文只给了 suspect 和脚注里的方差结论，雅可比端的完整论证是后来社区补齐的 [2,3]。

## 附录 B sinusoidal 位置编码的线性性质

**目标**：证明 PE(pos+k) 可由 PE(pos) 线性表示。

利用三角恒等式：

$$\sin(pos + k) = \sin(pos)\cos(k) + \cos(pos)\sin(k)$$

$$\cos(pos + k) = \cos(pos)\cos(k) - \sin(pos)\sin(k)$$

对固定频率 ω（对应某个维度对），位置编码的分量是 sin(ω·pos) 与 cos(ω·pos)。位置 pos+k 的分量：

$$\begin{bmatrix} \sin(\omega(pos+k)) \\ \cos(\omega(pos+k)) \end{bmatrix} = \begin{bmatrix} \cos(\omega k) & \sin(\omega k) \\ -\sin(\omega k) & \cos(\omega k) \end{bmatrix} \begin{bmatrix} \sin(\omega pos) \\ \cos(\omega pos) \end{bmatrix}$$

**关键性质**：变换矩阵只依赖偏移量 k 与频率 ω，不依赖绝对位置 pos。同一个线性变换（旋转矩阵）对任意位置通用——模型只要学会"这个旋转对应偏移 k"，就能在所有位置上识别同样的相对距离。

每个维度对独立做这样的旋转，整个 PE(pos+k) = R(k)·PE(pos)，R(k) 是块对角旋转矩阵。**相对位置关系被编码成了一族与绝对位置无关的线性变换**，这是 sinusoidal 方案最核心的设计收益。

## 附录 C 复杂度对照的推导细节

原论文 Table 1 三条主路线的账（n 序列长度、d 表示维度、k 卷积核宽、r 受限邻域半径）：

**Self-Attention**：QK^T 是 n×d_k 与 d_k×n 的矩阵乘，O(n²·d)；无递归依赖，串行操作 O(1)；任意两位置直连，路径 O(1)。

**Recurrent**：每步算 W_h·h（d×d 矩阵乘向量）+ W_x·x，n 步共 O(n·d²)；必须逐步算，串行 O(n)；位置 1 到位置 n 的信息要传 n−1 站，路径 O(n)。

**Convolutional**：每层 n 个位置各做 k×d² 的卷积，O(k·n·d²)；层间无依赖可并行，串行 O(1)；路径上，a 层卷积覆盖 1+a(k−1) 个输入，覆盖 n 需要约 log_{k}(n) 层，路径 O(log_k n)。

**restricted Self-Attention**：每个位置只和半径 r 的邻域交互，O(r·n·d)；串行 O(1)；跨出半径 r 需要逐层中转，路径 O(n/r)。

读表的方法：串行操作数决定训练速度（GPU 并行度），最大路径长度决定长程依赖的学习难度（梯度跨越的连乘次数）。Transformer 用 O(n²) 的计算量买了 O(1) 的路径——这笔交易在序列长度可控时（2017 年的翻译任务，n < 100）稳赚，在长文档、长上下文场景（n 上万）开始亏，这正是第八章死穴一的由来。

---

## 参考文献

| # | 来源 | 类型 | 链接 | 主要引用章节 |
|---|------|------|------|--------------|
| 1 | Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need. NIPS 2017.（原文已核对：PDF + arXiv 源码包，含 sqrt_d_trick.tex） | 论文原文 | https://arxiv.org/abs/1706.03762 | 全文；三（3.2.1 与 Footnote 4）、四（动机句）、五（对比实验）、六（Post-Norm）、七（Table 1） |
| 2 | 为什么注意力机制中要除以 √dk：从方差到梯度的推导 | 知乎专栏 | https://zhuanlan.zhihu.com/p/1964145108153799386 | 三、附录 A（方差+雅可比推导主源） |
| 3 | 一文搞懂 Transformer 中除以√d 的作用，为什么不是2√d | 知乎专栏 | https://zhuanlan.zhihu.com/p/32150751004 | 三（方差归一必然性） |
| 4 | The Illustrated Transformer（Jay Alammar） | 经典博客（图源，CC BY-NC-SA） | https://jalammar.github.io/illustrated-transformer/ | 二、四、五、六（全部 jalammar 配图） |
| 5 | Transformer升级之路：1、Sinusoidal位置编码追根溯源（苏剑林） | 知乎专栏 | https://zhuanlan.zhihu.com/p/359500899 | 五、附录 B |
| 6 | The Annotated Transformer（Harvard NLP） | 经典博客 | https://nlp.seas.harvard.edu/annotated-transformer/ | 六（mask 实现细节） |
| 7 | Attention? Attention!（Lilian Weng） | 经典博客 | https://lilianweng.github.io/posts/2018-06-24-attention/ | 二（注意力家族谱系） |
| 8 | 从《Convolutional Sequence to Sequence Learning》到《Attention Is All You Need》 | 知乎专栏 | https://zhuanlan.zhihu.com/p/27464080 | 七（时间线、对照分析、英法细节） |
| 9 | Gehring J, Auli M, Grangier D, et al. Convolutional Sequence to Sequence Learning. ICML 2017. | 论文原文（PDF 已下载） | https://arxiv.org/abs/1705.03122 | 七（ConvS2S 架构与 WMT 成绩） |
| 10 | 《注意力即是一切》：Transformer 诞生秘史 | 知乎专栏（访谈回忆类，文学化重构，仅叙事素材） | https://zhuanlan.zhihu.com/p/2057208655284590487 | 三（诺姆提出 √dk 的回忆）、六（Pre-Norm 错误的来源警示） |
| 11 | 从 NLP 到 LLM 系列（三）：RNN、LSTM 与 Seq2Seq（本站） | 博客 | /2026/09/05/llm-series-03-rnn-lstm-seq2seq/ | 一、二、六、九 |
| 12 | 从 NLP 到 LLM 系列（二）：词嵌入与 Word2Vec（本站） | 博客 | /2026/09/04/llm-series-02-word2vec/ | 七（可扩展性裁决模式） |
