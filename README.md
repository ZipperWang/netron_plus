# Netron Plus

一个用 **Rust + Tauri** 编写的 ONNX 模型查看与分析工具：查看模型结构图、算子和张量，并提供形状推断、FLOPs/参数量、内存估算、权重数据统计和模型健康检查。

> 本项目受 [Netron](https://github.com/lutzroeder/netron) 启发，是独立实现，与 Netron 及其作者无关。

## 功能

- **结构图**：Canvas 渲染 + Rust 实现的分层布局，上万节点也能流畅缩放；层次细节（LOD）、小地图、面包屑导航子图（If / Loop / Scan / 本地函数）、追踪上下游、按算子/模块/类别高亮、超长连线自动简化、导出 PNG / SVG
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

在 [Releases](../../releases) 或 [Actions](../../actions) 的构建产物中下载 Windows 安装包（`.exe` / `.msi`）。

## 项目结构

```
crates/core   Rust 核心：ONNX 解析（手写 prost 定义）、形状推断、计算量、内存、检查、图布局
crates/wasm   核心库的 WebAssembly 桥接（网页版使用）
src-tauri     Tauri 桌面程序
src           React 前端（Canvas 图渲染、表格、检查器、图表）
scripts       生成测试模型的脚本（CI 中运行）
```

所有计算都在 Rust 中完成；桌面版通过 Tauri IPC 调用，网页版在 Web Worker 中运行同一份 Rust 代码编译出的 WebAssembly。

## 构建

本仓库通过 GitHub Actions 构建（`.github/workflows/ci.yml`）：

- `core`：运行 Rust 单元测试，并用 ONNX 官方形状推断结果校验本项目的形状推断
- `web`：构建 WebAssembly 核心与前端
- `desktop`：在 Windows 上构建 Tauri 安装包

本地构建需要 Rust 和 Node.js 22：

```bash
npm install
npx tauri icon app-icon.png
npx tauri build
```

## English

Netron Plus is an ONNX model viewer and analyzer built with Rust and Tauri: architecture graph, operator and tensor tables, shape inference, FLOPs/parameter/memory estimates, weight statistics and model health checks. All analysis runs in Rust (natively in the desktop app, as WebAssembly in the web build).

## License

MIT
