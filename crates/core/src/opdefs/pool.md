# Pooling

<!-- Format: see the module documentation of crates/core/src/opfunc.rs. -->

## MaxPool
en: N-D max pooling; padded positions never win.
zh: N 维最大池化；填充位置不参与取最大值。
note.en: With ceil_mode = 1 the size is rounded up, but a last window that would start in the right padding is dropped.
note.zh: ceil_mode = 1 时向上取整，但若最后一个窗口的起点落在右侧填充区内则丢弃该窗口。
```tex
Y_{n,c,\mathbf{o}} = \max_{\mathbf{k}} X_{n,\, c,\; \mathbf{s} \odot \mathbf{o} + \mathbf{d} \odot \mathbf{k} - \mathbf{p}^{\mathrm{begin}}}
```
```tex shape
O_i = \floor{\frac{I_i + p^{\mathrm{begin}}_i + p^{\mathrm{end}}_i - d_i\,(K_i - 1) - 1}{s_i}} + 1 \qquad (\text{ceil\_mode} = 0)
```
```tex ceil_mode = 1
O'_i = \ceil{\frac{I_i + p^{\mathrm{begin}}_i + p^{\mathrm{end}}_i - d_i\,(K_i - 1) - 1}{s_i}} + 1, \qquad O_i = \begin{cases} O'_i - 1, & (O'_i - 1)\, s_i \ge I_i + p^{\mathrm{begin}}_i \\ O'_i, & \text{otherwise} \end{cases}
```
sym: \mathbf{K} = attr:kernel_shape
sym: \mathbf{s} = attr:strides ?= (1, \ldots, 1)
sym: \mathbf{d} = attr:dilations ?= (1, \ldots, 1)
sym: \mathbf{p} = attr:pads ?= (0, \ldots, 0)
sym: \text{auto\_pad} = attr:auto_pad ?= \texttt{NOTSET}
sym: \text{ceil\_mode} = attr:ceil_mode ?= 0
sym: \mathbf{I} = in:X.shape[2:]
sym: \mathbf{O} = out:Y.shape[2:]

## AveragePool
en: N-D average pooling.
zh: N 维平均池化。
note.en: count_include_pad = 0 divides by the number of non-padding elements in each window; 1 divides by the full window size.
note.zh: count_include_pad = 0 时除以窗口内非填充元素的个数；为 1 时除以完整窗口大小。
```tex
Y_{n,c,\mathbf{o}} = \frac{1}{\lvert \mathcal{K}_{\mathbf{o}} \rvert} \sum_{\mathbf{k} \in \mathcal{K}_{\mathbf{o}}} X_{n,\, c,\; \mathbf{s} \odot \mathbf{o} + \mathbf{d} \odot \mathbf{k} - \mathbf{p}^{\mathrm{begin}}}
```
```tex shape
O_i = \floor{\frac{I_i + p^{\mathrm{begin}}_i + p^{\mathrm{end}}_i - d_i\,(K_i - 1) - 1}{s_i}} + 1 \qquad (\text{ceil\_mode} = 0)
```
```tex ceil_mode = 1
O'_i = \ceil{\frac{I_i + p^{\mathrm{begin}}_i + p^{\mathrm{end}}_i - d_i\,(K_i - 1) - 1}{s_i}} + 1, \qquad O_i = \begin{cases} O'_i - 1, & (O'_i - 1)\, s_i \ge I_i + p^{\mathrm{begin}}_i \\ O'_i, & \text{otherwise} \end{cases}
```
sym: \mathbf{K} = attr:kernel_shape
sym: \mathbf{s} = attr:strides ?= (1, \ldots, 1)
sym: \mathbf{p} = attr:pads ?= (0, \ldots, 0)
sym: \text{count\_include\_pad} = attr:count_include_pad ?= 0
sym: \text{ceil\_mode} = attr:ceil_mode ?= 0
sym: \mathbf{I} = in:X.shape[2:]
sym: \mathbf{O} = out:Y.shape[2:]

## GlobalAveragePool
en: Average over all spatial dimensions.
zh: 在所有空间维度上取平均。
```tex
Y_{n,c,1,\ldots,1} = \frac{1}{\prod_i I_i} \sum_{\mathbf{i}} X_{n,c,\mathbf{i}}
```
sym: \mathbf{I} = in:X.shape[2:]

## GlobalMaxPool
en: Maximum over all spatial dimensions.
zh: 在所有空间维度上取最大值。
```tex
Y_{n,c,1,\ldots,1} = \max_{\mathbf{i}} X_{n,c,\mathbf{i}}
```
sym: \mathbf{I} = in:X.shape[2:]

## LpPool
en: N-D Lp pooling.
zh: N 维 Lp 池化。
```tex
Y_{n,c,\mathbf{o}} = \Bigl(\sum_{\mathbf{k}} \bigl\lvert X_{n,\, c,\; \mathbf{s} \odot \mathbf{o} + \mathbf{d} \odot \mathbf{k} - \mathbf{p}^{\mathrm{begin}}} \bigr\rvert^{p}\Bigr)^{1/p}
```
sym: p = attr:p ?= 2
sym: \mathbf{K} = attr:kernel_shape
sym: \mathbf{s} = attr:strides ?= (1, \ldots, 1)
