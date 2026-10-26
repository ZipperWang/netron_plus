# Normalization

<!-- Format: see the module documentation of crates/core/src/opfunc.rs. -->

## BatchNormalization
en: Batch normalization with running statistics, per channel (dimension 1).
zh: 批归一化，在推理模式下使用滑动统计量，按通道（第 1 维）计算。
note.en: In training mode (training_mode = 1, opset 14+) the batch statistics are used and running_mean / running_var are updated with momentum.
note.zh: 训练模式（training_mode = 1，opset 14 起）使用当前批次统计量，并按 momentum 更新 running_mean / running_var。
```tex
Y_{n,c,\ldots} = \gamma_c \, \frac{X_{n,c,\ldots} - \mu_c}{\sqrt{\sigma_c^{2} + \epsilon}} + \beta_c
```
sym: \epsilon = attr:epsilon ?= 10^{-5}
sym: \text{momentum} = attr:momentum ?= 0.9
sym: C = in:X.shape[1]
sym: \operatorname{shape}(X) = in:X.shape

## LayerNormalization
since: 17
en: Layer normalization over the trailing dimensions starting at axis.
zh: 层归一化，对从 axis 开始的所有尾部维度做归一化。
```tex
Y = \frac{X - \mu}{\sqrt{\sigma^{2} + \epsilon}} \odot \text{Scale} + B, \qquad \mu = \frac{1}{D} \sum_{i \in \mathcal{A}} X_i, \quad \sigma^{2} = \frac{1}{D} \sum_{i \in \mathcal{A}} \left(X_i - \mu\right)^{2}
```
```tex where
\mathcal{A} = \{\text{axis}, \ldots, r - 1\}, \qquad D = \prod_{k \in \mathcal{A}} d_k
```
sym: \text{axis} = attr:axis ?= -1
sym: \epsilon = attr:epsilon ?= 10^{-5}
sym: r = in:X.rank
sym: (d_0, \ldots) = in:X.shape

## InstanceNormalization
en: Instance normalization: per sample and channel over the spatial dimensions.
zh: 实例归一化：对每个样本、每个通道在空间维度上归一化。
```tex
Y_{n,c,\mathbf{i}} = \gamma_c \, \frac{X_{n,c,\mathbf{i}} - \mu_{n,c}}{\sqrt{\sigma_{n,c}^{2} + \epsilon}} + \beta_c, \qquad \mu_{n,c} = \operatorname{E}_{\mathbf{i}}\left[X_{n,c,\mathbf{i}}\right]
```
sym: \epsilon = attr:epsilon ?= 10^{-5}
sym: C = in:input.shape[1]

## GroupNormalization
since: 18
en: Group normalization; scale and bias have one value per group (opset 18–20).
zh: 分组归一化；scale 与 bias 每组一个值（opset 18–20）。
note.en: Opset 21 changed scale and bias to one value per channel.
note.zh: opset 21 起 scale 与 bias 改为每通道一个值。
```tex
Y_{n,c,\mathbf{i}} = \gamma_{g} \, \frac{X_{n,c,\mathbf{i}} - \mu_{n,g}}{\sqrt{\sigma_{n,g}^{2} + \epsilon}} + \beta_{g}, \qquad g = \floor{\frac{c}{C/G}}
```
sym: G = attr:num_groups
sym: \epsilon = attr:epsilon ?= 10^{-5}
sym: C = in:X.shape[1]

## GroupNormalization
since: 21
en: Group normalization: statistics per group of C/G channels, affine per channel.
zh: 分组归一化：每组 C/G 个通道计算统计量，按通道做仿射变换。
```tex
Y_{n,c,\mathbf{i}} = \gamma_{c} \, \frac{X_{n,c,\mathbf{i}} - \mu_{n,g}}{\sqrt{\sigma_{n,g}^{2} + \epsilon}} + \beta_{c}, \qquad g = \floor{\frac{c}{C/G}}
```
sym: G = attr:num_groups
sym: \epsilon = attr:epsilon ?= 10^{-5}
sym: C = in:X.shape[1]

## RMSNormalization
since: 23
en: Root mean square normalization over the trailing dimensions starting at axis.
zh: 均方根归一化，对从 axis 开始的尾部维度计算。
```tex
Y = \frac{X}{\sqrt{\frac{1}{D} \sum_{i \in \mathcal{A}} X_i^{2} + \epsilon}} \odot \text{scale}, \qquad \mathcal{A} = \{\text{axis}, \ldots, r - 1\}
```
sym: \text{axis} = attr:axis ?= -1
sym: \epsilon = attr:epsilon ?= 10^{-5}
sym: \operatorname{shape}(X) = in:X.shape

## LpNormalization
en: Divides by the Lp norm along an axis.
zh: 沿指定轴除以 Lp 范数。
```tex
Y = \frac{X}{\lVert X \rVert_p}, \qquad \lVert X \rVert_p = \Bigl(\sum_{j} \lvert X_j \rvert^{p}\Bigr)^{1/p}
```
sym: \text{axis} = attr:axis ?= -1
sym: p = attr:p ?= 2

## MeanVarianceNormalization
en: Normalizes to zero mean and unit variance over the given axes.
zh: 在给定轴上归一化为零均值、单位方差。
```tex
Y = \frac{X - \operatorname{E}[X]}{\sqrt{\operatorname{E}\left[X^{2}\right] - \operatorname{E}[X]^{2}} + 10^{-9}}
```
sym: \text{axes} = attr:axes ?= (0,\, 2,\, 3)

## LRN
en: Local response normalization across neighbouring channels.
zh: 局部响应归一化，在相邻通道间计算。
```tex
Y_{n,c,\mathbf{i}} = \frac{X_{n,c,\mathbf{i}}}{\left(\text{bias} + \frac{\alpha}{\text{size}} \sum_{c' = \max\left(0,\, c - \floor{\frac{\text{size} - 1}{2}}\right)}^{\min\left(C - 1,\, c + \ceil{\frac{\text{size} - 1}{2}}\right)} X_{n,c',\mathbf{i}}^{2}\right)^{\beta}}
```
sym: \text{size} = attr:size
sym: \alpha = attr:alpha ?= 10^{-4}
sym: \beta = attr:beta ?= 0.75
sym: \text{bias} = attr:bias ?= 1
sym: C = in:X.shape[1]
