# ONNX Runtime contrib operators (com.microsoft)

<!-- Format: see the module documentation of crates/core/src/opfunc.rs. -->
<!-- Contrib ops without an entry here fall back to the standard definition. -->

## com.microsoft::Gelu
en: Gaussian error linear unit (exact, erf form).
zh: 高斯误差线性单元（精确的 erf 形式）。
```tex
Y = \frac{X}{2}\left(1 + \erf\left(\frac{X}{\sqrt{2}}\right)\right)
```

## com.microsoft::FastGelu
en: Tanh approximation of GELU with an optional bias added first.
zh: GELU 的 tanh 近似，可先加偏置。
```tex
Y = \frac{X'}{2}\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(X' + 0.044715\, X'^{3}\right)\right)\right), \qquad X' = X + \text{bias}
```

## com.microsoft::BiasGelu
en: Exact GELU of A + B.
zh: 对 A + B 计算精确 GELU。
```tex
C = \frac{A + B}{2}\left(1 + \erf\left(\frac{A + B}{\sqrt{2}}\right)\right)
```

## com.microsoft::QuickGelu
en: Sigmoid approximation of GELU.
zh: GELU 的 sigmoid 近似。
```tex
Y = X \cdot \sigma(\alpha X)
```
sym: \alpha = attr:alpha ?= 1.702

## com.microsoft::BiasAdd
en: Adds a bias and a skip input.
zh: 加上偏置和残差输入。
```tex
Y = X + \text{bias} + \text{skip}
```

## com.microsoft::BiasSoftmax
en: Softmax of data plus a broadcast bias.
zh: 对 data 加上广播的 bias 后做 softmax。
```tex
\text{output} = \softmax_{\text{axis}}\left(\text{data} + \text{bias}\right)
```
sym: \text{axis} = attr:axis ?= 1

## com.microsoft::SkipLayerNormalization
en: Residual add followed by layer normalization over the last dimension.
zh: 残差相加后在最后一维做层归一化。
```tex
S = \text{input} + \text{skip} + \text{bias}, \qquad \text{output} = \frac{S - \mu}{\sqrt{\sigma^{2} + \epsilon}} \odot \gamma + \beta
```
sym: \epsilon = attr:epsilon
sym: \operatorname{shape}(\text{input}) = in:input.shape

## com.microsoft::SimplifiedLayerNormalization
en: RMS normalization over the trailing dimensions starting at axis.
zh: 从 axis 开始的尾部维度上的 RMS 归一化。
```tex
Y = \frac{X}{\sqrt{\frac{1}{D} \sum_{i} X_i^{2} + \epsilon}} \odot \text{scale}
```
sym: \text{axis} = attr:axis ?= -1
sym: \epsilon = attr:epsilon ?= 10^{-5}
sym: \operatorname{shape}(X) = in:X.shape

## com.microsoft::SkipSimplifiedLayerNormalization
en: Residual add followed by RMS normalization over the last dimension.
zh: 残差相加后在最后一维做 RMS 归一化。
```tex
S = \text{input} + \text{skip} + \text{bias}, \qquad \text{output} = \frac{S}{\sqrt{\frac{1}{D} \sum_{i} S_i^{2} + \epsilon}} \odot \gamma
```
sym: \epsilon = attr:epsilon
sym: \operatorname{shape}(\text{input}) = in:input.shape

## com.microsoft::FusedMatMul
en: Matrix product with optional transposes and scaling.
zh: 可选转置并缩放的矩阵乘法。
```tex
Y = \alpha \cdot \operatorname{op}_{\text{transA}}(A)\, \operatorname{op}_{\text{transB}}(B)
```
sym: \alpha = attr:alpha ?= 1
sym: \text{transA} = attr:transA ?= 0
sym: \text{transB} = attr:transB ?= 0
sym: \operatorname{shape}(A) = in:A.shape
sym: \operatorname{shape}(B) = in:B.shape

## com.microsoft::FusedConv
en: Convolution fused with a residual input and an activation.
zh: 融合了残差输入和激活函数的卷积。
```tex
Y = \operatorname{act}\left(\operatorname{conv}(X, W) + B + Z\right)
```
sym: \operatorname{act} = attr:activation
sym: \mathbf{K} = attr:kernel_shape, in:W.shape[2:]
sym: \mathbf{s} = attr:strides ?= (1, \ldots, 1)
sym: \mathbf{p} = attr:pads ?= (0, \ldots, 0)

## com.microsoft::MatMulNBits
en: Matrix product with a block-wise N-bit quantized B, stored transposed (N × K).
zh: B 为按块 N 比特量化、转置存储（N × K）的矩阵乘法。
```tex
Y = A\, \hat{B}^{\top}, \qquad \hat{B}_{n,k} = \left(B_{n,k} - z_{n,\floor{k/g}}\right) s_{n,\floor{k/g}}
```
sym: K = attr:K
sym: N = attr:N
sym: \text{bits} = attr:bits ?= 4
sym: g = attr:block_size
sym: \operatorname{shape}(A) = in:A.shape

## com.microsoft::Attention
en: Multi-head self-attention with a fused QKV projection.
zh: 融合 QKV 投影的多头自注意力。
```tex
[Q, K, V] = \text{input} \cdot W + b, \qquad \text{output} = \softmax\left(\text{scale} \cdot Q K^{\top} + M\right) V, \qquad \text{scale} = \frac{1}{\sqrt{d}} \text{ by default}
```
sym: h = attr:num_heads
sym: \text{unidirectional} = attr:unidirectional ?= 0
sym: \text{scale} = attr:scale ?= 1 / \sqrt{d}
sym: \operatorname{shape}(\text{input}) = in:input.shape

## com.microsoft::MultiHeadAttention
en: Multi-head attention on separate query, key and value inputs.
zh: 分别输入 query、key、value 的多头注意力。
```tex
\text{output} = \concat_{h}\left(\softmax\left(\text{scale} \cdot Q_h K_h^{\top} + M\right) V_h\right)
```
sym: h = attr:num_heads
sym: \text{scale} = attr:scale ?= 1 / \sqrt{d}
sym: \operatorname{shape}(Q) = in:query.shape

## com.microsoft::GroupQueryAttention
en: Grouped-query attention with KV cache; each KV head serves several query heads.
zh: 带 KV 缓存的分组查询注意力；每个 KV 头服务多个 Q 头。
```tex
\text{output}_h = \softmax\left(\text{scale} \cdot Q_h K_{\floor{h / r}}^{\top} + M\right) V_{\floor{h / r}}, \qquad r = \frac{\text{num\_heads}}{\text{kv\_num\_heads}}
```
sym: \text{num\_heads} = attr:num_heads
sym: \text{kv\_num\_heads} = attr:kv_num_heads
sym: \text{scale} = attr:scale ?= 1 / \sqrt{d}
sym: \operatorname{shape}(Q) = in:query.shape
