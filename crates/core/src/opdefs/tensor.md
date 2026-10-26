# Shape, indexing and data movement

<!-- Format: see the module documentation of crates/core/src/opfunc.rs. -->

## Reshape
en: Reshapes the data; 0 copies the input dimension (allowzero = 0) and one -1 is inferred.
zh: 改变形状；shape 中的 0 表示沿用输入维度（allowzero = 0），一个 -1 由元素总数推出。
```tex
Y = \operatorname{reshape}\left(X,\, \text{shape}\right), \qquad \prod_i \operatorname{shape}(Y)_i = \prod_i \operatorname{shape}(X)_i
```
sym: \text{shape} = in:shape.value
sym: \text{allowzero} = attr:allowzero ?= 0
sym: \operatorname{shape}(X) = in:data.shape
sym: \operatorname{shape}(Y) = out:reshaped.shape

## Transpose
en: Permutes the axes; output axis k is input axis perm[k].
zh: 按 perm 重排维度；输出第 k 维对应输入第 perm[k] 维。
```tex
Y_{j_0, \ldots, j_{r-1}} = X_{i_0, \ldots, i_{r-1}}, \quad i_{\pi_k} = j_k, \qquad \operatorname{shape}(Y)_k = \operatorname{shape}(X)_{\pi_k}
```
sym: \pi = attr:perm ?= (r-1, \ldots, 1, 0)
sym: \operatorname{shape}(X) = in:data.shape
sym: \operatorname{shape}(Y) = out:transposed.shape

## Concat
en: Concatenates the inputs along an axis.
zh: 沿指定轴拼接所有输入。
```tex
Y = \concat_{\text{axis}}\left(X_0, \ldots, X_{n-1}\right), \qquad \operatorname{shape}(Y)_{\text{axis}} = \sum_{k} \operatorname{shape}(X_k)_{\text{axis}}
```
sym: \text{axis} = attr:axis
sym: \operatorname{shape}(Y) = out:concat_result.shape

## Split
en: Splits the input along an axis into pieces of the given sizes (equal when not given).
zh: 沿指定轴按给定长度切分（未给出时等分）。
```tex
Y_k = X_{\ldots,\; o_k : o_k + \ell_k,\; \ldots}, \qquad o_k = \sum_{j < k} \ell_j
```
sym: \text{axis} = attr:axis ?= 0
sym: \boldsymbol{\ell} = in:split.value, attr:split ?= \text{equal parts}
sym: \text{num\_outputs} = attr:num_outputs
sym: \operatorname{shape}(X) = in:input.shape

## Slice
en: Takes a strided slice along the given axes, like X[starts:ends:steps].
zh: 沿给定轴取带步长的切片，相当于 X[starts:ends:steps]。
note.en: Before opset 10 starts, ends and axes are attributes; ends are clamped to the dimension size.
note.zh: opset 10 之前 starts、ends、axes 为属性；超出范围的 ends 会被截断到维度大小。
```tex
Y = X_{[\, b_a : e_a : t_a \,]_{a \in \text{axes}}}, \qquad \operatorname{shape}(Y)_a = \max\left(0,\, \ceil{\frac{e_a - b_a}{t_a}}\right)
```
sym: \mathbf{b} = in:starts.value, attr:starts
sym: \mathbf{e} = in:ends.value, attr:ends
sym: \text{axes} = in:axes.value, attr:axes ?= (0, \ldots, r-1)
sym: \mathbf{t} = in:steps.value ?= (1, \ldots, 1)
sym: \operatorname{shape}(X) = in:data.shape
sym: \operatorname{shape}(Y) = out:output.shape

## Gather
en: Gathers slices along an axis using the indices (negative indices count from the end).
zh: 按 indices 沿指定轴收集切片（负索引从末尾计数）。
```tex
Y_{\mathbf{a},\, \mathbf{j},\, \mathbf{b}} = X_{\mathbf{a},\; \text{indices}_{\mathbf{j}},\; \mathbf{b}}, \qquad \operatorname{rank}(Y) = q + r - 1
```
sym: \text{axis} = attr:axis ?= 0
sym: \text{indices} = in:indices.value
sym: \operatorname{shape}(X) = in:data.shape
sym: \operatorname{shape}(\text{indices}) = in:indices.shape

## GatherElements
en: Gathers single elements along an axis; indices has the same rank as data.
zh: 沿指定轴逐元素收集；indices 与 data 的秩相同。
```tex
Y_{i_0, \ldots, i_{r-1}} = X_{i_0, \ldots,\; \text{indices}_{i_0, \ldots, i_{r-1}},\; \ldots, i_{r-1}} \quad (\text{index replaced at axis})
```
sym: \text{axis} = attr:axis ?= 0

## GatherND
en: Gathers slices addressed by the last dimension of indices.
zh: 以 indices 最后一维作为多维坐标收集切片。
```tex
Y_{\mathbf{b},\, \mathbf{i}} = X_{\mathbf{b},\; \text{indices}_{\mathbf{b},\, \mathbf{i},\, :},\; \ldots}
```
sym: \text{batch\_dims} = attr:batch_dims ?= 0
sym: \operatorname{shape}(\text{indices}) = in:indices.shape

## ScatterND
en: Copies data and writes updates at the positions given by indices.
zh: 复制 data，并在 indices 指定的位置写入 updates。
```tex
\text{output} = \text{data}, \qquad \text{output}\left[\text{indices}_{\mathbf{i}}\right] \leftarrow \text{updates}_{\mathbf{i}}
```
sym: \text{reduction} = attr:reduction ?= \texttt{none}

## ScatterElements
en: Copies data and writes each update along an axis at its index.
zh: 复制 data，并沿指定轴把每个 update 写到对应索引处。
```tex
\text{output}_{\ldots,\; \text{indices}_{\mathbf{i}},\; \ldots} \leftarrow \text{updates}_{\mathbf{i}} \quad (\text{index replaced at axis})
```
sym: \text{axis} = attr:axis ?= 0
sym: \text{reduction} = attr:reduction ?= \texttt{none}

## Squeeze
en: Removes dimensions of size 1 (all of them when axes is empty).
zh: 删除长度为 1 的维度（未给出 axes 时删除全部）。
```tex
\operatorname{shape}(Y) = \operatorname{shape}(X) \setminus \{\, d_a = 1 : a \in \text{axes} \,\}
```
sym: \text{axes} = in:axes.value, attr:axes ?= \text{all size-1 axes}
sym: \operatorname{shape}(X) = in:data.shape
sym: \operatorname{shape}(Y) = out:squeezed.shape

## Unsqueeze
en: Inserts dimensions of size 1 at the given output axes.
zh: 在给定的输出位置插入长度为 1 的维度。
```tex
\operatorname{rank}(Y) = \operatorname{rank}(X) + \lvert \text{axes} \rvert, \qquad \operatorname{shape}(Y)_a = 1 \ \ (a \in \text{axes})
```
sym: \text{axes} = in:axes.value, attr:axes
sym: \operatorname{shape}(X) = in:data.shape
sym: \operatorname{shape}(Y) = out:expanded.shape

## Flatten
en: Flattens into a 2-D matrix split at axis.
zh: 在 axis 处分开，展平成二维矩阵。
```tex
Y = \operatorname{reshape}\left(X,\, \Bigl(\prod_{k < \text{axis}} d_k,\ \prod_{k \ge \text{axis}} d_k\Bigr)\right)
```
sym: \text{axis} = attr:axis ?= 1
sym: (d_0, \ldots) = in:input.shape

## Expand
en: Broadcasts the input to the given shape.
zh: 把输入广播到给定形状。
```tex
Y = \operatorname{broadcast}\left(X,\, \text{shape}\right)
```
sym: \text{shape} = in:shape.value
sym: \operatorname{shape}(X) = in:input.shape
sym: \operatorname{shape}(Y) = out:output.shape

## Tile
en: Repeats the input along every axis.
zh: 沿每个维度重复输入。
```tex
Y_{\mathbf{i}} = X_{\mathbf{i} \bmod \operatorname{shape}(X)}, \qquad \operatorname{shape}(Y)_k = \operatorname{shape}(X)_k \cdot \text{repeats}_k
```
sym: \text{repeats} = in:repeats.value
sym: \operatorname{shape}(X) = in:input.shape

## Pad
en: Pads the input; the mode decides the padded values.
zh: 对输入做填充；mode 决定填充值。
note.en: pads is [x1_begin, x2_begin, ..., x1_end, x2_end, ...]; negative pads crop.
note.zh: pads 的排列为 [x1_begin, x2_begin, …, x1_end, x2_end, …]；负值表示裁剪。
```tex
Y_{\mathbf{i}} = \begin{cases} X_{\mathbf{i} - \mathbf{p}^{\mathrm{begin}}}, & \text{inside } X \\ c, & \texttt{constant} \\ X \text{ mirrored at the border}, & \texttt{reflect} \\ X \text{ at the nearest border}, & \texttt{edge} \\ X \text{ wrapped around}, & \texttt{wrap} \end{cases}
```
```tex shape
O_k = I_k + p^{\mathrm{begin}}_k + p^{\mathrm{end}}_k
```
sym: \text{mode} = attr:mode ?= \texttt{constant}
sym: \mathbf{p} = in:pads.value, attr:pads
sym: c = in:constant_value.value, attr:value ?= 0
sym: \mathbf{I} = in:data.shape
sym: \mathbf{O} = out:output.shape

## Resize
since: 10
en: Resizes by scale factors with nearest or linear interpolation (asymmetric coordinates).
zh: 按比例缩放，使用最近邻或线性插值（asymmetric 坐标变换）。
note.en: Opset 11 added sizes, roi, cubic mode and the coordinate transformation modes.
note.zh: opset 11 起新增 sizes、roi、cubic 模式和多种坐标变换方式。
```tex
Y_{\mathbf{x}} = \operatorname{interp}_{\text{mode}}\left(X,\, \frac{\mathbf{x}}{\text{scales}}\right), \qquad O_i = \floor{I_i \cdot \text{scales}_i}
```
sym: \text{mode} = attr:mode ?= \texttt{nearest}
sym: \text{scales} = in:1.value
sym: \mathbf{I} = in:X.shape
sym: \mathbf{O} = out:Y.shape

## Resize
since: 11
en: Resizes with nearest, linear or cubic interpolation.
zh: 使用最近邻、线性或三次插值缩放。
note.en: The output size is floor(I × scale) unless sizes is given; nearest_mode decides how x_in is rounded in nearest mode.
note.zh: 未给出 sizes 时输出尺寸为 ⌊I × scale⌋；最近邻模式下 nearest_mode 决定 x_in 的取整方式。
```tex
Y_{\mathbf{x}} = \operatorname{interp}_{\text{mode}}\left(X,\, x_{\mathrm{in}}(\mathbf{x})\right), \qquad O_i = \floor{I_i \cdot \text{scale}_i}
```
```tex coordinate_transformation_mode
x_{\mathrm{in}} = \begin{cases} \dfrac{x + 0.5}{\text{scale}} - 0.5, & \texttt{half\_pixel} \\ \dfrac{x + 0.5}{\text{scale}} - 0.5 \ \ (O > 1), \ \ 0 \ \ (O = 1), & \texttt{pytorch\_half\_pixel} \\ x \, \dfrac{I - 1}{O - 1}, & \texttt{align\_corners} \\ \dfrac{x}{\text{scale}}, & \texttt{asymmetric} \end{cases}
```
sym: \text{mode} = attr:mode ?= \texttt{nearest}
sym: \text{coordinate\_transformation\_mode} = attr:coordinate_transformation_mode ?= \texttt{half\_pixel}
sym: \text{nearest\_mode} = attr:nearest_mode ?= \texttt{round\_prefer\_floor}
sym: a = attr:cubic_coeff_a ?= -0.75
sym: \text{scale} = in:scales.value
sym: \text{sizes} = in:sizes.value
sym: \mathbf{I} = in:X.shape
sym: \mathbf{O} = out:Y.shape

## Upsample
en: Deprecated resize by scale factors (nearest or linear, asymmetric coordinates).
zh: 已弃用的按比例缩放（最近邻或线性，asymmetric 坐标变换）。
```tex
Y_{\mathbf{x}} = \operatorname{interp}_{\text{mode}}\left(X,\, \frac{\mathbf{x}}{\text{scales}}\right), \qquad O_i = \floor{I_i \cdot \text{scales}_i}
```
sym: \text{mode} = attr:mode ?= \texttt{nearest}
sym: \text{scales} = in:scales.value, attr:scales
sym: \mathbf{I} = in:X.shape
sym: \mathbf{O} = out:Y.shape

## Identity
en: Returns the input unchanged.
zh: 原样输出输入。
```tex
Y = X
```

## Cast
en: Converts the element type.
zh: 转换元素数据类型。
note.en: Float to integer conversion truncates toward zero; out-of-range results are undefined unless saturate applies (float8 targets).
note.zh: 浮点转整数向零截断；超出范围的结果未定义（float8 目标类型可用 saturate 控制饱和）。
```tex
Y = \operatorname{cast}_{\text{to}}(X)
```
sym: \text{to} = dtype:to
sym: \operatorname{type}(X) = in:input.dtype

## CastLike
en: Converts the element type to that of target_type.
zh: 把元素类型转换为 target_type 的类型。
```tex
Y = \operatorname{cast}_{\operatorname{type}(\text{target\_type})}(X)
```
sym: \operatorname{type}(\text{target\_type}) = in:target_type.dtype

## Shape
en: The shape of the input as a 1-D int64 tensor, optionally sliced.
zh: 以一维 int64 张量输出输入的形状，可切片。
```tex
Y = \operatorname{shape}(X)_{[\text{start} : \text{end}]}
```
sym: \text{start} = attr:start ?= 0
sym: \text{end} = attr:end ?= r
sym: \operatorname{shape}(X) = in:data.shape

## Size
en: The number of elements of the input.
zh: 输入的元素总数。
```tex
Y = \prod_{k} d_k
```
sym: (d_0, \ldots) = in:data.shape

## ConstantOfShape
en: A tensor of the given shape filled with value.
zh: 生成给定形状、填充 value 的张量。
```tex
Y = \text{value} \cdot \mathbf{1}_{\text{shape}}
```
sym: \text{value} = attr:value ?= 0
sym: \text{shape} = in:input.value

## Range
en: Evenly spaced values in [start, limit).
zh: 生成 [start, limit) 内的等差数列。
```tex
Y_i = \text{start} + i \cdot \text{delta}, \qquad 0 \le i < \max\left(\ceil{\frac{\text{limit} - \text{start}}{\text{delta}}},\, 0\right)
```
sym: \text{start} = in:start.value
sym: \text{limit} = in:limit.value
sym: \text{delta} = in:delta.value

## DepthToSpace
en: Moves channel blocks into spatial blocks.
zh: 把通道维中的块重排到空间维。
```tex mode = DCR
Y_{n,\, c,\, h b + i,\, w b + j} = X_{n,\; (i b + j)\frac{C}{b^{2}} + c,\; h,\; w}
```
```tex mode = CRD
Y_{n,\, c,\, h b + i,\, w b + j} = X_{n,\; c b^{2} + i b + j,\; h,\; w}
```
sym: b = attr:blocksize
sym: \text{mode} = attr:mode ?= \texttt{DCR}
sym: C = in:input.shape[1]

## SpaceToDepth
en: Moves spatial blocks into the channel dimension.
zh: 把空间维中的块重排到通道维。
```tex
Y_{n,\; (i b + j) C + c,\; h,\; w} = X_{n,\, c,\, h b + i,\, w b + j}
```
sym: b = attr:blocksize
sym: C = in:input.shape[1]

## Trilu
en: Keeps the upper or lower triangle of the last two dimensions, offset by k.
zh: 保留最后两维的上三角或下三角部分，偏移量为 k。
```tex upper = 1
Y_{\ldots,i,j} = \begin{cases} X_{\ldots,i,j}, & j - i \ge k \\ 0, & \text{otherwise} \end{cases}
```
```tex upper = 0
Y_{\ldots,i,j} = \begin{cases} X_{\ldots,i,j}, & j - i \le k \\ 0, & \text{otherwise} \end{cases}
```
sym: \text{upper} = attr:upper ?= 1
sym: k = in:k.value ?= 0

## NonZero
en: Indices of the non-zero elements, one column per element.
zh: 非零元素的坐标，每个元素占一列。
```tex
Y = \left[\, \mathbf{i} : X_{\mathbf{i}} \ne 0 \,\right] \in \mathbb{Z}^{r \times n_{\ne 0}}
```
sym: r = in:X.rank

## NonMaxSuppression
en: Greedy non-maximum suppression per batch and class.
zh: 按批次和类别做贪心非极大值抑制。
note.en: selected_indices rows are (batch, class, box); center_point_box = 1 means boxes are (x, y, w, h).
note.zh: selected_indices 每行为 (batch, class, box)；center_point_box = 1 表示框格式为 (x, y, w, h)。
```tex
\operatorname{IoU}(b_i, b_j) = \frac{\lvert b_i \cap b_j \rvert}{\lvert b_i \cup b_j \rvert}, \qquad \text{keep } b \text{ if } s_b > \text{score\_threshold} \text{ and } \operatorname{IoU}(b, b') \le \text{iou\_threshold} \ \forall \text{ kept } b'
```
sym: \text{max\_output\_boxes\_per\_class} = in:max_output_boxes_per_class.value ?= 0
sym: \text{iou\_threshold} = in:iou_threshold.value ?= 0
sym: \text{score\_threshold} = in:score_threshold.value
sym: \text{center\_point\_box} = attr:center_point_box ?= 0
sym: \operatorname{shape}(\text{boxes}) = in:boxes.shape
