# Reductions

<!-- Format: see the module documentation of crates/core/src/opfunc.rs. -->
<!-- Opset 18 (ReduceSum: 13) moved axes from an attribute to an input. -->

## ReduceSum
en: Sum over the given axes (all axes when empty).
zh: 在给定轴上求和（未给出时对所有轴）。
```tex
Y = \sum_{\mathcal{A}} X
```
sym: \mathcal{A} = in:axes.value, attr:axes ?= \text{all axes}
sym: \text{keepdims} = attr:keepdims ?= 1
sym: \text{noop\_with\_empty\_axes} = attr:noop_with_empty_axes ?= 0
sym: \operatorname{shape}(X) = in:data.shape

## ReduceMean
en: Mean over the given axes (all axes when empty).
zh: 在给定轴上求平均（未给出时对所有轴）。
```tex
Y = \frac{1}{N_{\mathcal{A}}} \sum_{\mathcal{A}} X, \qquad N_{\mathcal{A}} = \prod_{a \in \mathcal{A}} d_a
```
sym: \mathcal{A} = in:axes.value, attr:axes ?= \text{all axes}
sym: \text{keepdims} = attr:keepdims ?= 1
sym: (d_0, \ldots) = in:data.shape

## ReduceMax
en: Maximum over the given axes.
zh: 在给定轴上取最大值。
```tex
Y = \max_{\mathcal{A}} X
```
sym: \mathcal{A} = in:axes.value, attr:axes ?= \text{all axes}
sym: \text{keepdims} = attr:keepdims ?= 1

## ReduceMin
en: Minimum over the given axes.
zh: 在给定轴上取最小值。
```tex
Y = \min_{\mathcal{A}} X
```
sym: \mathcal{A} = in:axes.value, attr:axes ?= \text{all axes}
sym: \text{keepdims} = attr:keepdims ?= 1

## ReduceProd
en: Product over the given axes.
zh: 在给定轴上求积。
```tex
Y = \prod_{\mathcal{A}} X
```
sym: \mathcal{A} = in:axes.value, attr:axes ?= \text{all axes}
sym: \text{keepdims} = attr:keepdims ?= 1

## ReduceL1
en: L1 norm over the given axes.
zh: 在给定轴上求 L1 范数。
```tex
Y = \sum_{\mathcal{A}} \lvert X \rvert
```
sym: \mathcal{A} = in:axes.value, attr:axes ?= \text{all axes}
sym: \text{keepdims} = attr:keepdims ?= 1

## ReduceL2
en: L2 norm over the given axes.
zh: 在给定轴上求 L2 范数。
```tex
Y = \sqrt{\sum_{\mathcal{A}} X^{2}}
```
sym: \mathcal{A} = in:axes.value, attr:axes ?= \text{all axes}
sym: \text{keepdims} = attr:keepdims ?= 1

## ReduceSumSquare
en: Sum of squares over the given axes.
zh: 在给定轴上求平方和。
```tex
Y = \sum_{\mathcal{A}} X^{2}
```
sym: \mathcal{A} = in:axes.value, attr:axes ?= \text{all axes}
sym: \text{keepdims} = attr:keepdims ?= 1

## ReduceLogSum
en: Logarithm of the sum over the given axes.
zh: 在给定轴上求和后取对数。
```tex
Y = \ln \sum_{\mathcal{A}} X
```
sym: \mathcal{A} = in:axes.value, attr:axes ?= \text{all axes}
sym: \text{keepdims} = attr:keepdims ?= 1

## ReduceLogSumExp
en: Log-sum-exp over the given axes.
zh: 在给定轴上计算 log-sum-exp。
```tex
Y = \ln \sum_{\mathcal{A}} e^{X}
```
sym: \mathcal{A} = in:axes.value, attr:axes ?= \text{all axes}
sym: \text{keepdims} = attr:keepdims ?= 1

## ArgMax
en: Index of the maximum along an axis.
zh: 沿指定轴取最大值的索引。
note.en: select_last_index = 1 returns the last index when the maximum appears several times.
note.zh: select_last_index = 1 时，最大值重复出现会返回最后一个索引。
```tex
Y_{\ldots} = \argmax_{i} X_{\ldots,\, i,\, \ldots}
```
sym: \text{axis} = attr:axis ?= 0
sym: \text{keepdims} = attr:keepdims ?= 1
sym: \text{select\_last\_index} = attr:select_last_index ?= 0

## ArgMin
en: Index of the minimum along an axis.
zh: 沿指定轴取最小值的索引。
```tex
Y_{\ldots} = \argmin_{i} X_{\ldots,\, i,\, \ldots}
```
sym: \text{axis} = attr:axis ?= 0
sym: \text{keepdims} = attr:keepdims ?= 1
sym: \text{select\_last\_index} = attr:select_last_index ?= 0

## TopK
en: The K largest (or smallest) elements along an axis and their indices.
zh: 沿指定轴取最大（或最小）的 K 个元素及其索引。
```tex
(\text{Values}, \text{Indices}) = \operatorname{topk}_{K}\left(X,\, \text{axis}\right), \qquad \operatorname{shape}(\text{Values})_{\text{axis}} = K
```
sym: K = in:K.value
sym: \text{axis} = attr:axis ?= -1
sym: \text{largest} = attr:largest ?= 1
sym: \text{sorted} = attr:sorted ?= 1
sym: \operatorname{shape}(X) = in:X.shape
