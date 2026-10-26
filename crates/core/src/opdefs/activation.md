# Activation functions

<!-- Format: see the module documentation of crates/core/src/opfunc.rs. -->

## Relu
en: Rectified linear unit, applied element-wise.
zh: 线性整流函数，逐元素计算。
```tex
Y = \max(0,\, X)
```

## LeakyRelu
en: Leaky ReLU: negative inputs are scaled by alpha.
zh: 带泄漏的 ReLU：负输入乘以 alpha。
```tex
Y = \begin{cases} X, & X \ge 0 \\ \alpha X, & X < 0 \end{cases}
```
sym: \alpha = attr:alpha ?= 0.01

## PRelu
en: Parametric ReLU: negative inputs are scaled by a learned slope (broadcast to X).
zh: 参数化 ReLU：负输入乘以可学习的 slope（单向广播到 X）。
```tex
Y = \begin{cases} X, & X \ge 0 \\ \text{slope} \cdot X, & X < 0 \end{cases}
```
sym: \operatorname{shape}(\text{slope}) = in:slope.shape

## Sigmoid
en: Logistic sigmoid, applied element-wise.
zh: Logistic sigmoid 函数，逐元素计算。
```tex
Y = \sigma(X) = \frac{1}{1 + e^{-X}}
```

## HardSigmoid
en: Piece-wise linear approximation of the sigmoid.
zh: sigmoid 的分段线性近似。
```tex
Y = \max\left(0,\, \min\left(1,\, \alpha X + \beta\right)\right)
```
sym: \alpha = attr:alpha ?= 0.2
sym: \beta = attr:beta ?= 0.5

## HardSwish
en: Hard swish: X times a hard sigmoid with alpha = 1/6, beta = 0.5.
zh: Hard swish：X 乘以 alpha = 1/6、beta = 0.5 的 HardSigmoid。
```tex
Y = X \cdot \max\left(0,\, \min\left(1,\, \frac{X}{6} + \frac{1}{2}\right)\right)
```

## Tanh
en: Hyperbolic tangent, applied element-wise.
zh: 双曲正切函数，逐元素计算。
```tex
Y = \tanh(X) = \frac{e^{X} - e^{-X}}{e^{X} + e^{-X}}
```

## Elu
en: Exponential linear unit.
zh: 指数线性单元。
```tex
Y = \begin{cases} X, & X \ge 0 \\ \alpha\left(e^{X} - 1\right), & X < 0 \end{cases}
```
sym: \alpha = attr:alpha ?= 1

## Selu
en: Scaled exponential linear unit.
zh: 缩放指数线性单元。
```tex
Y = \gamma \begin{cases} X, & X > 0 \\ \alpha\left(e^{X} - 1\right), & X \le 0 \end{cases}
```
sym: \alpha = attr:alpha ?= 1.67326
sym: \gamma = attr:gamma ?= 1.0507

## Celu
en: Continuously differentiable exponential linear unit.
zh: 连续可微的指数线性单元。
```tex
Y = \max(0,\, X) + \min\left(0,\, \alpha\left(e^{X / \alpha} - 1\right)\right)
```
sym: \alpha = attr:alpha ?= 1

## Gelu
since: 20
en: Gaussian error linear unit, exact (erf) or tanh approximation.
zh: 高斯误差线性单元，精确形式（erf）或 tanh 近似。
```tex approximate = none
Y = \frac{X}{2}\left(1 + \erf\left(\frac{X}{\sqrt{2}}\right)\right)
```
```tex approximate = tanh
Y = \frac{X}{2}\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(X + 0.044715\, X^{3}\right)\right)\right)
```
sym: \text{approximate} = attr:approximate ?= \texttt{none}

## Mish
en: Mish: X times tanh of softplus(X).
zh: Mish：X 乘以 softplus(X) 的双曲正切。
```tex
Y = X \tanh\left(\ln\left(1 + e^{X}\right)\right)
```

## Softplus
en: Smooth approximation of ReLU.
zh: ReLU 的平滑近似。
```tex
Y = \ln\left(1 + e^{X}\right)
```

## Softsign
en: Softsign, applied element-wise.
zh: Softsign 函数，逐元素计算。
```tex
Y = \frac{X}{1 + \lvert X \rvert}
```

## Swish
en: Swish (SiLU when alpha = 1).
zh: Swish 函数（alpha = 1 时即 SiLU）。
```tex
Y = X \cdot \sigma(\alpha X) = \frac{X}{1 + e^{-\alpha X}}
```
sym: \alpha = attr:alpha ?= 1

## ThresholdedRelu
en: Passes values above alpha, zero elsewhere.
zh: 大于 alpha 的值原样输出，其余为 0。
```tex
Y = \begin{cases} X, & X > \alpha \\ 0, & \text{otherwise} \end{cases}
```
sym: \alpha = attr:alpha ?= 1

## Shrink
en: Shrinks values towards zero by bias outside [-lambd, lambd], zero inside.
zh: 在 [-lambd, lambd] 之外向 0 收缩 bias，区间内为 0。
```tex
Y = \begin{cases} X + b, & X < -\lambda \\ X - b, & X > \lambda \\ 0, & \text{otherwise} \end{cases}
```
sym: \lambda = attr:lambd ?= 0.5
sym: b = attr:bias ?= 0

## Clip
since: 6
en: Limits values to [min, max]; bounds are attributes before opset 11.
zh: 将数值限制在 [min, max]；opset 11 之前上下界为属性。
```tex
Y = \min\left(\max\left(X,\, \text{min}\right),\, \text{max}\right)
```
sym: \text{min} = attr:min ?= -3.4028\times10^{38}
sym: \text{max} = attr:max ?= 3.4028\times10^{38}

## Clip
since: 11
en: Limits values to [min, max]; missing bounds are unbounded.
zh: 将数值限制在 [min, max]；未提供的边界视为无界。
```tex
Y = \min\left(\max\left(X,\, \text{min}\right),\, \text{max}\right)
```
sym: \text{min} = in:min.value ?= -\infty
sym: \text{max} = in:max.value ?= +\infty

## Softmax
since: 1
en: Softmax over the input coerced to 2-D at axis (opset < 13).
zh: 先在 axis 处把输入压成二维，再按行做 softmax（opset < 13）。
note.en: Opset 13 changed the semantics to a softmax along a single axis.
note.zh: opset 13 起语义改为沿单个轴做 softmax，导出时需注意版本差异。
```tex
Y_{n,i} = \frac{e^{X_{n,i}}}{\sum_{j=0}^{D-1} e^{X_{n,j}}}, \qquad N = \prod_{k < \text{axis}} d_k,\quad D = \prod_{k \ge \text{axis}} d_k
```
sym: \text{axis} = attr:axis ?= 1
sym: (d_0, \ldots) = in:input.shape

## Softmax
since: 13
en: Softmax along one axis.
zh: 沿指定轴做 softmax。
note.en: Implementations subtract the maximum along the axis first for numerical stability.
note.zh: 实现中通常先减去该轴上的最大值以保证数值稳定。
```tex
Y_{\ldots, i, \ldots} = \frac{e^{X_{\ldots, i, \ldots}}}{\sum_{j} e^{X_{\ldots, j, \ldots}}} \qquad (i, j \text{ index the axis})
```
sym: \text{axis} = attr:axis ?= -1
sym: (d_0, \ldots) = in:input.shape

## LogSoftmax
since: 13
en: Logarithm of the softmax along one axis.
zh: 沿指定轴计算 softmax 的对数。
```tex
Y_{\ldots, i, \ldots} = X_{\ldots, i, \ldots} - \ln \sum_{j} e^{X_{\ldots, j, \ldots}}
```
sym: \text{axis} = attr:axis ?= -1

## Hardmax
since: 13
en: One-hot of the first maximum along one axis.
zh: 沿指定轴取第一个最大值位置为 1、其余为 0。
```tex
Y_{\ldots, i, \ldots} = \begin{cases} 1, & i = \argmax_{j} X_{\ldots, j, \ldots} \\ 0, & \text{otherwise} \end{cases}
```
sym: \text{axis} = attr:axis ?= -1
