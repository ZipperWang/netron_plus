# 构建约定

- 本项目统一使用GitHub Actions安装依赖、编译和测试。
- 本地仅编辑源码及执行无需下载依赖的静态检查，不执行npm install、cargo fetch、cargo build、cargo test或Tauri构建等命令。
- 构建流程位于.github/workflows/ci.yml和.github/workflows/release.yml，Windows交付产物为单个免安装程序netron-plus.exe。
- 验证构建结果时查看GitHub Actions的运行状态、日志和产物，不在本地重新安装构建依赖。
