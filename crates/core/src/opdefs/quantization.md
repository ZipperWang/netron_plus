# Quantization

<!-- Format: see the module documentation of crates/core/src/opfunc.rs. -->

## QuantizeLinear
en: Linear quantization; scale and zero point are per tensor, per axis or per block.
zh: 线性量化；scale 与 zero point 可按张量、按轴或按块给出。
note.en: Rounding is half to even; saturation clamps to the range of the output type, e.g. [0, 255] for uint8.
note.zh: 取整为四舍六入五成双；饱和截断到输出类型的范围，例如 uint8 为 [0, 255]。
```tex
y = \sat\left(\round\left(\frac{x}{y_{\text{scale}}}\right) + y_{\text{zero\_point}}\right)
```
sym: y_{\text{scale}} = in:y_scale.value
sym: y_{\text{zero\_point}} = in:y_zero_point.value ?= 0
sym: \text{axis} = attr:axis ?= 1
sym: \text{block\_size} = attr:block_size ?= 0
sym: \operatorname{type}(y) = out:y.dtype, in:y_zero_point.dtype ?= \texttt{uint8}

## DequantizeLinear
en: Linear dequantization; scale and zero point are per tensor, per axis or per block.
zh: 线性反量化；scale 与 zero point 可按张量、按轴或按块给出。
```tex
y = \left(x - x_{\text{zero\_point}}\right) \cdot x_{\text{scale}}
```
sym: x_{\text{scale}} = in:x_scale.value
sym: x_{\text{zero\_point}} = in:x_zero_point.value ?= 0
sym: \text{axis} = attr:axis ?= 1
sym: \operatorname{type}(x) = in:x.dtype

## DynamicQuantizeLinear
en: Quantizes to uint8 with scale and zero point computed from the data range.
zh: 根据数据范围计算 scale 与 zero point，量化为 uint8。
```tex
\begin{aligned}
y_{\text{scale}} &= \frac{\max(0, \max x) - \min(0, \min x)}{255} \\
y_{\text{zero\_point}} &= \sat\left(\round\left(-\frac{\min(0, \min x)}{y_{\text{scale}}}\right)\right) \\
y &= \sat\left(\round\left(\frac{x}{y_{\text{scale}}}\right) + y_{\text{zero\_point}}\right)
\end{aligned}
```

## QLinearConv
en: Quantized convolution: dequantize, convolve, requantize.
zh: 量化卷积：反量化、卷积、再量化。
note.en: B is int32 with scale x_scale × w_scale and zero point 0.
note.zh: B 为 int32，其 scale 为 x_scale × w_scale，zero point 为 0。
```tex
y = \sat\left(\round\left(\frac{x_s\, w_s}{y_s} \left(\operatorname{conv}\left(x - x_{zp},\, w - w_{zp}\right) + B\right)\right) + y_{zp}\right)
```
sym: x_s = in:x_scale.value
sym: w_s = in:w_scale.value
sym: y_s = in:y_scale.value
sym: x_{zp} = in:x_zero_point.value
sym: w_{zp} = in:w_zero_point.value
sym: y_{zp} = in:y_zero_point.value
sym: \mathbf{K} = attr:kernel_shape, in:w.shape[2:]
sym: \mathbf{s} = attr:strides ?= (1, \ldots, 1)
sym: \mathbf{p} = attr:pads ?= (0, \ldots, 0)
sym: G = attr:group ?= 1

## QLinearMatMul
en: Quantized matrix product: dequantize, multiply, requantize.
zh: 量化矩阵乘法：反量化、相乘、再量化。
```tex
y = \sat\left(\round\left(\frac{a_s\, b_s}{y_s} \left(a - a_{zp}\right)\left(b - b_{zp}\right)\right) + y_{zp}\right)
```
sym: a_s = in:a_scale.value
sym: b_s = in:b_scale.value
sym: y_s = in:y_scale.value
sym: a_{zp} = in:a_zero_point.value
sym: b_{zp} = in:b_zero_point.value
sym: y_{zp} = in:y_zero_point.value

## MatMulInteger
en: Integer matrix product with zero points, accumulated in int32.
zh: 带 zero point 的整数矩阵乘法，以 int32 累加。
```tex
Y = \left(A - a_{zp}\right)\left(B - b_{zp}\right)
```
sym: a_{zp} = in:a_zero_point.value ?= 0
sym: b_{zp} = in:b_zero_point.value ?= 0

## ConvInteger
en: Integer convolution with zero points, accumulated in int32.
zh: 带 zero point 的整数卷积，以 int32 累加。
```tex
y = \operatorname{conv}\left(x - x_{zp},\, w - w_{zp}\right)
```
sym: x_{zp} = in:x_zero_point.value ?= 0
sym: w_{zp} = in:w_zero_point.value ?= 0
