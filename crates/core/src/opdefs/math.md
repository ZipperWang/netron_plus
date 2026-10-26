# Element-wise math and logic

<!-- Format: see the module documentation of crates/core/src/opfunc.rs. -->

## Add
en: Element-wise addition with multidirectional (NumPy) broadcasting.
zh: 逐元素加法，支持多向（NumPy 规则）广播。
```tex
C = A + B
```
sym: \operatorname{shape}(A) = in:A.shape
sym: \operatorname{shape}(B) = in:B.shape
sym: \operatorname{shape}(C) = out:C.shape

## Sub
en: Element-wise subtraction with multidirectional broadcasting.
zh: 逐元素减法，支持多向广播。
```tex
C = A - B
```
sym: \operatorname{shape}(A) = in:A.shape
sym: \operatorname{shape}(B) = in:B.shape

## Mul
en: Element-wise multiplication with multidirectional broadcasting.
zh: 逐元素乘法，支持多向广播。
```tex
C = A \odot B
```
sym: \operatorname{shape}(A) = in:A.shape
sym: \operatorname{shape}(B) = in:B.shape

## Div
en: Element-wise division with multidirectional broadcasting.
zh: 逐元素除法，支持多向广播。
note.en: Integer division truncates toward zero.
note.zh: 整数类型的除法向零取整。
```tex
C = \frac{A}{B}
```
sym: \operatorname{shape}(A) = in:A.shape
sym: \operatorname{shape}(B) = in:B.shape

## Pow
en: Element-wise power with multidirectional broadcasting.
zh: 逐元素幂运算，支持多向广播。
```tex
Z = X^{Y}
```
sym: Y = in:Y.value

## Sqrt
en: Element-wise square root.
zh: 逐元素平方根。
```tex
Y = \sqrt{X}
```

## Exp
en: Element-wise natural exponential.
zh: 逐元素自然指数。
```tex
Y = e^{X}
```

## Log
en: Element-wise natural logarithm.
zh: 逐元素自然对数。
```tex
Y = \ln X
```

## Abs
en: Element-wise absolute value.
zh: 逐元素绝对值。
```tex
Y = \lvert X \rvert
```

## Neg
en: Element-wise negation.
zh: 逐元素取负。
```tex
Y = -X
```

## Reciprocal
en: Element-wise reciprocal.
zh: 逐元素取倒数。
```tex
Y = \frac{1}{X}
```

## Floor
en: Element-wise floor.
zh: 逐元素向下取整。
```tex
Y = \floor{X}
```

## Ceil
en: Element-wise ceiling.
zh: 逐元素向上取整。
```tex
Y = \ceil{X}
```

## Round
en: Element-wise rounding; halves round to the nearest even integer.
zh: 逐元素四舍五入，恰为 .5 时取最近的偶数（银行家舍入）。
```tex
Y = \round(X), \qquad \round(2.5) = 2,\ \round(3.5) = 4
```

## Sign
en: Element-wise sign.
zh: 逐元素符号函数。
```tex
Y = \begin{cases} 1, & X > 0 \\ 0, & X = 0 \\ -1, & X < 0 \end{cases}
```

## Erf
en: Element-wise Gauss error function.
zh: 逐元素高斯误差函数。
```tex
Y = \erf(X) = \frac{2}{\sqrt{\pi}} \int_{0}^{X} e^{-t^{2}}\, dt
```

## Sin
en: Element-wise sine.
zh: 逐元素正弦。
```tex
Y = \sin(X)
```

## Cos
en: Element-wise cosine.
zh: 逐元素余弦。
```tex
Y = \cos(X)
```

## Max
en: Element-wise maximum of all inputs, with broadcasting.
zh: 所有输入的逐元素最大值，支持广播。
```tex
Y = \max\left(X_0,\, X_1,\, \ldots,\, X_{n-1}\right)
```

## Min
en: Element-wise minimum of all inputs, with broadcasting.
zh: 所有输入的逐元素最小值，支持广播。
```tex
Y = \min\left(X_0,\, X_1,\, \ldots,\, X_{n-1}\right)
```

## Sum
en: Element-wise sum of all inputs, with broadcasting.
zh: 所有输入的逐元素求和，支持广播。
```tex
Y = \sum_{k=0}^{n-1} X_k
```

## Mean
en: Element-wise mean of all inputs, with broadcasting.
zh: 所有输入的逐元素平均，支持广播。
```tex
Y = \frac{1}{n} \sum_{k=0}^{n-1} X_k
```

## Mod
en: Element-wise remainder; fmod selects the sign convention.
zh: 逐元素取余；fmod 决定结果符号的约定。
```tex fmod = 0
C = A - B \floor{\frac{A}{B}} \qquad \text{(sign of the divisor, like Python)}
```
```tex fmod = 1
C = A - B \cdot \operatorname{trunc}\left(\frac{A}{B}\right) \qquad \text{(sign of the dividend, like C fmod)}
```
sym: \text{fmod} = attr:fmod ?= 0

## Where
en: Selects elements from X or Y depending on condition, with broadcasting.
zh: 按 condition 逐元素从 X 或 Y 中选取，支持广播。
```tex
\text{output} = \begin{cases} X, & \text{condition} \\ Y, & \text{otherwise} \end{cases}
```

## Equal
en: Element-wise equality test.
zh: 逐元素判断是否相等。
```tex
C = \begin{cases} \text{true}, & A = B \\ \text{false}, & \text{otherwise} \end{cases}
```

## Greater
en: Element-wise A > B.
zh: 逐元素判断 A > B。
```tex
C = \begin{cases} \text{true}, & A > B \\ \text{false}, & \text{otherwise} \end{cases}
```

## GreaterOrEqual
en: Element-wise A >= B.
zh: 逐元素判断 A ≥ B。
```tex
C = \begin{cases} \text{true}, & A \ge B \\ \text{false}, & \text{otherwise} \end{cases}
```

## Less
en: Element-wise A < B.
zh: 逐元素判断 A < B。
```tex
C = \begin{cases} \text{true}, & A < B \\ \text{false}, & \text{otherwise} \end{cases}
```

## LessOrEqual
en: Element-wise A <= B.
zh: 逐元素判断 A ≤ B。
```tex
C = \begin{cases} \text{true}, & A \le B \\ \text{false}, & \text{otherwise} \end{cases}
```

## Not
en: Element-wise logical negation.
zh: 逐元素逻辑非。
```tex
Y = \lnot X
```

## And
en: Element-wise logical and.
zh: 逐元素逻辑与。
```tex
C = A \land B
```

## Or
en: Element-wise logical or.
zh: 逐元素逻辑或。
```tex
C = A \lor B
```

## Xor
en: Element-wise logical exclusive or.
zh: 逐元素逻辑异或。
```tex
C = A \oplus B
```

## IsNaN
en: Tests each element for NaN.
zh: 逐元素判断是否为 NaN。
```tex
Y = \left[\, X \ne X \,\right]
```

## CumSum
en: Cumulative sum along an axis.
zh: 沿指定轴的累加和。
```tex exclusive = 0, reverse = 0
y_{i} = \sum_{k=0}^{i} x_{k}
```
```tex exclusive = 1
y_{i} = \sum_{k=0}^{i-1} x_{k}
```
```tex reverse = 1
y_{i} = \sum_{k=i}^{n-1} x_{k}
```
sym: \text{axis} = in:axis.value
sym: \text{exclusive} = attr:exclusive ?= 0
sym: \text{reverse} = attr:reverse ?= 0

## Einsum
en: Einstein summation over the inputs as written in the equation.
zh: 按 equation 描述的爱因斯坦求和约定计算。
```tex
Y = \operatorname{einsum}\left(\text{equation},\, X_0, \ldots, X_{n-1}\right), \quad \text{e.g. } \texttt{ij,jk->ik}:\ Y_{ik} = \sum_{j} X_{0,ij}\, X_{1,jk}
```
sym: \text{equation} = attr:equation
