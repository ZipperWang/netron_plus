# Convolution, matrix products, recurrent and attention layers

<!-- Format: see the module documentation of crates/core/src/opfunc.rs. -->

## Conv
en: N-D convolution with optional bias and groups.
zh: N 维卷积，可带偏置和分组。
note.en: pads is [x1_begin, x2_begin, ..., x1_end, x2_end, ...]; with auto_pad SAME_UPPER / SAME_LOWER the output size is ceil(I / s).
note.zh: pads 的排列为 [x1_begin, x2_begin, …, x1_end, x2_end, …]；auto_pad 为 SAME_UPPER / SAME_LOWER 时输出尺寸为 ⌈I/s⌉，padding 自动补齐。
```tex
Y_{n,\, m,\, \mathbf{o}} = B_m + \sum_{c=0}^{C/G-1} \sum_{\mathbf{k}} X_{n,\; g\frac{C}{G} + c,\; \mathbf{s} \odot \mathbf{o} + \mathbf{d} \odot \mathbf{k} - \mathbf{p}^{\mathrm{begin}}} \, W_{m,\, c,\, \mathbf{k}}, \qquad g = \floor{\frac{m}{M/G}}
```
```tex shape
O_i = \floor{\frac{I_i + p^{\mathrm{begin}}_i + p^{\mathrm{end}}_i - d_i\,(K_i - 1) - 1}{s_i}} + 1
```
sym: \operatorname{shape}(X) = in:X.shape
sym: \operatorname{shape}(W) = in:W.shape
sym: C = in:X.shape[1]
sym: M = in:W.shape[0]
sym: G = attr:group ?= 1
sym: \mathbf{K} = attr:kernel_shape, in:W.shape[2:]
sym: \mathbf{s} = attr:strides ?= (1, \ldots, 1)
sym: \mathbf{d} = attr:dilations ?= (1, \ldots, 1)
sym: \mathbf{p} = attr:pads ?= (0, \ldots, 0)
sym: \text{auto\_pad} = attr:auto_pad ?= \texttt{NOTSET}
sym: \mathbf{I} = in:X.shape[2:]
sym: \mathbf{O} = out:Y.shape[2:]

## ConvTranspose
en: N-D transposed convolution (gradient of Conv), with optional bias and groups.
zh: N 维转置卷积（Conv 的梯度运算），可带偏置和分组。
note.en: W has shape (C, M/G, k1, k2, ...). When output_shape is given, the pads are computed from it.
note.zh: W 的形状为 (C, M/G, k1, k2, …)；给出 output_shape 时按它反推 pads。
```tex
Y_{n,\; g\frac{M}{G} + m,\; \mathbf{o}} = B_{g M/G + m} + \sum_{c=0}^{C/G-1} \sum_{\substack{\mathbf{i}, \mathbf{k} \\ \mathbf{s} \odot \mathbf{i} + \mathbf{d} \odot \mathbf{k} - \mathbf{p}^{\mathrm{begin}} = \mathbf{o}}} X_{n,\; g\frac{C}{G} + c,\; \mathbf{i}} \, W_{g\frac{C}{G} + c,\; m,\; \mathbf{k}}
```
```tex shape
O_i = s_i\,(I_i - 1) + a_i + d_i\,(K_i - 1) + 1 - p^{\mathrm{begin}}_i - p^{\mathrm{end}}_i
```
sym: \operatorname{shape}(X) = in:X.shape
sym: \operatorname{shape}(W) = in:W.shape
sym: C = in:X.shape[1]
sym: M/G = in:W.shape[1]
sym: G = attr:group ?= 1
sym: \mathbf{K} = attr:kernel_shape, in:W.shape[2:]
sym: \mathbf{s} = attr:strides ?= (1, \ldots, 1)
sym: \mathbf{d} = attr:dilations ?= (1, \ldots, 1)
sym: \mathbf{p} = attr:pads ?= (0, \ldots, 0)
sym: \mathbf{a} = attr:output_padding ?= (0, \ldots, 0)
sym: \text{output\_shape} = attr:output_shape
sym: \mathbf{O} = out:Y.shape[2:]

## Gemm
en: General matrix multiplication with optional transposes and bias.
zh: 通用矩阵乘法，可选转置并加偏置。
```tex
Y = \alpha\, A' B' + \beta\, C, \qquad A' = \begin{cases} A^{\top}, & \text{transA} = 1 \\ A, & \text{otherwise} \end{cases} \quad B' = \begin{cases} B^{\top}, & \text{transB} = 1 \\ B, & \text{otherwise} \end{cases}
```
```tex shape
A' \in \mathbb{R}^{M \times K}, \quad B' \in \mathbb{R}^{K \times N}, \quad Y \in \mathbb{R}^{M \times N}
```
sym: \alpha = attr:alpha ?= 1
sym: \beta = attr:beta ?= 1
sym: \text{transA} = attr:transA ?= 0
sym: \text{transB} = attr:transB ?= 0
sym: \operatorname{shape}(A) = in:A.shape
sym: \operatorname{shape}(B) = in:B.shape
sym: \operatorname{shape}(C) = in:C.shape
sym: \operatorname{shape}(Y) = out:Y.shape

## MatMul
en: Matrix product with NumPy semantics; leading dimensions broadcast as a batch.
zh: NumPy 语义的矩阵乘法，前面的维度按批次广播。
```tex
Y_{\ldots,\, i,\, j} = \sum_{k=0}^{K-1} A_{\ldots,\, i,\, k}\, B_{\ldots,\, k,\, j}
```
sym: \operatorname{shape}(A) = in:A.shape
sym: \operatorname{shape}(B) = in:B.shape
sym: K = in:A.shape[-1]
sym: \operatorname{shape}(Y) = out:Y.shape

## RNN
en: Simple recurrent layer; f defaults to Tanh.
zh: 简单循环神经网络层；f 默认为 Tanh。
```tex
H_t = f\left(X_t W^{\top} + H_{t-1} R^{\top} + W_b + R_b\right)
```
sym: \text{hidden\_size} = attr:hidden_size
sym: f = attr:activations ?= \texttt{Tanh}
sym: \text{direction} = attr:direction ?= \texttt{forward}
sym: \operatorname{shape}(X) = in:X.shape

## LSTM
en: Long short-term memory layer (gate order i, o, f, c in W, R and B).
zh: 长短期记忆网络层（W、R、B 中门的顺序为 i、o、f、c）。
note.en: Defaults: f = Sigmoid, g = Tanh, h = Tanh; layout 0 means X is (seq, batch, input).
note.zh: 默认 f = Sigmoid、g = Tanh、h = Tanh；layout 为 0 时 X 的形状为 (seq, batch, input)。
```tex
\begin{aligned}
i_t &= f\left(X_t W_i^{\top} + H_{t-1} R_i^{\top} + P_i \odot C_{t-1} + W_{bi} + R_{bi}\right) \\
f_t &= f\left(X_t W_f^{\top} + H_{t-1} R_f^{\top} + P_f \odot C_{t-1} + W_{bf} + R_{bf}\right) \\
c_t &= g\left(X_t W_c^{\top} + H_{t-1} R_c^{\top} + W_{bc} + R_{bc}\right) \\
C_t &= f_t \odot C_{t-1} + i_t \odot c_t \\
o_t &= f\left(X_t W_o^{\top} + H_{t-1} R_o^{\top} + P_o \odot C_t + W_{bo} + R_{bo}\right) \\
H_t &= o_t \odot h\left(C_t\right)
\end{aligned}
```
sym: \text{hidden\_size} = attr:hidden_size
sym: (f, g, h) = attr:activations ?= (\texttt{Sigmoid},\, \texttt{Tanh},\, \texttt{Tanh})
sym: \text{direction} = attr:direction ?= \texttt{forward}
sym: \text{layout} = attr:layout ?= 0
sym: \operatorname{shape}(X) = in:X.shape
sym: \operatorname{shape}(W) = in:W.shape

## GRU
en: Gated recurrent unit layer (gate order z, r, h in W, R and B).
zh: 门控循环单元层（W、R、B 中门的顺序为 z、r、h）。
note.en: Defaults: f = Sigmoid, g = Tanh.
note.zh: 默认 f = Sigmoid、g = Tanh。
```tex
\begin{aligned}
z_t &= f\left(X_t W_z^{\top} + H_{t-1} R_z^{\top} + W_{bz} + R_{bz}\right) \\
r_t &= f\left(X_t W_r^{\top} + H_{t-1} R_r^{\top} + W_{br} + R_{br}\right) \\
H_t &= (1 - z_t) \odot h_t + z_t \odot H_{t-1}
\end{aligned}
```
```tex linear_before_reset = 0
h_t = g\left(X_t W_h^{\top} + \left(r_t \odot H_{t-1}\right) R_h^{\top} + R_{bh} + W_{bh}\right)
```
```tex linear_before_reset = 1
h_t = g\left(X_t W_h^{\top} + r_t \odot \left(H_{t-1} R_h^{\top} + R_{bh}\right) + W_{bh}\right)
```
sym: \text{hidden\_size} = attr:hidden_size
sym: \text{linear\_before\_reset} = attr:linear_before_reset ?= 0
sym: (f, g) = attr:activations ?= (\texttt{Sigmoid},\, \texttt{Tanh})
sym: \text{direction} = attr:direction ?= \texttt{forward}
sym: \operatorname{shape}(X) = in:X.shape

## Attention
since: 23
en: Scaled dot-product attention with optional mask, causal masking and KV cache.
zh: 缩放点积注意力，可带掩码、因果掩码和 KV 缓存。
note.en: With fewer KV heads than query heads (GQA), each KV head is shared by a group of query heads.
note.zh: KV 头数少于 Q 头数时（GQA），每个 KV 头被一组 Q 头共享。
```tex
Y = \softmax\left(\text{scale} \cdot Q K^{\top} + M\right) V, \qquad \text{scale} = \frac{1}{\sqrt{d_k}} \text{ by default}
```
```tex is_causal = 1
M_{ij} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}
```
sym: \text{scale} = attr:scale ?= 1 / \sqrt{d_k}
sym: \text{is\_causal} = attr:is_causal ?= 0
sym: \text{q\_num\_heads} = attr:q_num_heads
sym: \text{kv\_num\_heads} = attr:kv_num_heads
sym: \operatorname{shape}(Q) = in:Q.shape
sym: \operatorname{shape}(K) = in:K.shape

## RotaryEmbedding
since: 23
en: Rotary position embedding: rotates feature pairs by position-dependent angles.
zh: 旋转位置编码：按位置相关的角度旋转特征对。
```tex interleaved = 0
\begin{aligned}
y_{j} &= x_{j} \cos\theta_{p,j} - x_{j + d/2} \sin\theta_{p,j} \\
y_{j + d/2} &= x_{j + d/2} \cos\theta_{p,j} + x_{j} \sin\theta_{p,j}
\end{aligned} \qquad 0 \le j < d/2
```
```tex interleaved = 1
\begin{aligned}
y_{2j} &= x_{2j} \cos\theta_{p,j} - x_{2j+1} \sin\theta_{p,j} \\
y_{2j+1} &= x_{2j+1} \cos\theta_{p,j} + x_{2j} \sin\theta_{p,j}
\end{aligned}
```
sym: \text{interleaved} = attr:interleaved ?= 0
sym: d = attr:rotary_embedding_dim ?= \text{head size}
sym: \operatorname{shape}(X) = in:X.shape
sym: \operatorname{shape}(\cos) = in:cos_cache.shape
