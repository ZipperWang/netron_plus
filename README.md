# Netron Plus

一个用 **Rust + Tauri** 编写的 ONNX 模型查看与分析工具：查看模型结构图、算子和张量，并提供形状推断、FLOPs/参数量、内存估算、权重数据统计和模型健康检查。

> 本项目受 [Netron](https://github.com/lutzroeder/netron) 启发，是独立实现，与 Netron 及其作者无关。

## 功能

- **结构图**：Canvas 渲染 + Rust 实现的分层布局，上万节点也能流畅缩放；层次细节（LOD）、小地图、面包屑导航子图（If / Loop / Scan / 本地函数）、追踪上下游、按算子/模块/类别高亮、超长连线自动简化、导出 PNG / SVG
- **算子函数**：选中节点即可查看算子的数学定义（KaTeX 渲染的 TeX 公式、输出形状规则），公式中的符号自动代入本节点的属性、默认值、形状和常量输入；覆盖约 150 个常用标准算子与 ONNX Runtime contrib 算子，可复制 LaTeX；调用模型本地函数的节点可双击进入函数体
- **形状推断**：覆盖 150+ 常见算子，能解析 `Shape → Gather → Concat → Reshape` 这类导出模式；动态维度（如 `batch`、`seq`）可在概览页填入具体数值，实时重新计算
- **统计**
  - 参数量、MACs / FLOPs（逐节点、按算子类型、按模块）、激活内存总量与峰值
  - 算子分布、类别分布、权重数据类型分布、最大的张量
  - 节点表、张量表（排序、筛选、导出 CSV）
  - 权重统计：最小/最大值、均值、标准差、稀疏度、NaN/Inf、FP16 溢出风险
  - 单个张量：直方图、分位数、数值分页浏览，导出 `.npy` / `.csv` / 原始字节
- **健康检查**：未定义输入、环、拓扑顺序、未使用的初始化器/输入、死节点、重复名称、缺失 opset、已弃用算子、外部数据文件缺失、权重 NaN/Inf、FP16 溢出等
- 中英文界面、浅色/深色主题、常用快捷键（`Ctrl+O` 打开、`Ctrl+F` 搜索、`F` 适应屏幕、`↑/↓` 跳到上下游节点）

## 下载

在 [Releases](../../releases) 下载 `netron-plus.exe`，或在 [Actions](../../actions) 下载 `netron-plus-windows` 构建产物并解压，双击 `netron-plus.exe` 即可运行，无需安装本程序。

程序依赖系统的 Microsoft Edge WebView2 Runtime；缺少该运行时的电脑需要先安装 [WebView2 Runtime](https://developer.microsoft.com/microsoft-edge/webview2/)。

## 项目结构

```
crates/core   Rust 核心：ONNX 解析（手写 prost 定义）、形状推断、计算量、内存、检查、图布局
crates/wasm   核心库的 WebAssembly 桥接（网页版使用）
src-tauri     Tauri 桌面程序
src           React 前端（Canvas 图渲染、表格、检查器、图表）
scripts       生成测试模型的脚本（CI 中运行）
```

所有计算都在 Rust 中完成；桌面版通过 Tauri IPC 调用，网页版在 Web Worker 中运行同一份 Rust 代码编译出的 WebAssembly。

## 云端构建

本仓库统一通过 GitHub Actions 安装依赖、测试和编译，本地只保留源码，不安装构建依赖。

推送到 `main`、创建或更新 Pull Request 时会自动运行 CI（`.github/workflows/ci.yml`）：

- `core`：运行 Rust 单元测试，并用 ONNX 官方形状推断结果校验本项目的形状推断
- `web`：构建 WebAssembly 核心与前端
- `desktop`：在 Windows 上构建单个免安装程序 `netron-plus.exe`

需要手动构建时，在 [Actions](../../actions) 中选择 `CI` 或 `Release`，点击 `Run workflow`。完成后进入该次运行，在 Artifacts 中下载 `netron-plus-windows`，解压后得到单个 `netron-plus.exe`。

推送 `v*` 标签时会运行 `.github/workflows/release.yml`，并将 `netron-plus.exe` 上传到对应的 [Release](../../releases)。手动运行 `Release` 工作流时，可从该次运行的 Artifacts 下载程序。

Windows 程序在云端的生成路径为 `target/release/netron-plus.exe`。前端资源已嵌入程序，无需附带 `dist` 目录。

## English

Netron Plus is an ONNX model viewer and analyzer built with Rust and Tauri: architecture graph, operator and tensor tables, operator formulas rendered from TeX with the node's attributes and shapes filled in, shape inference, FLOPs/parameter/memory estimates, weight statistics and model health checks. All analysis runs in Rust (natively in the desktop app, as WebAssembly in the web build).

## License

MIT
