# Repository Guidelines

## Project Structure & Module Organization
- `Common/`: shared CUDA helpers used across kernels.
- `RegularUnits/`: baseline GPU microbenchmarks (ALU latency, cache and memory bandwidth/latency) with per-folder `Makefile` and `.cu`.
- `NewFeatures/`: DPX/TMA/DSM feature tests; each subdir has its own `Makefile` and sometimes `run_all.sh`.
- `TensorCores/`: mma/mmasp/wgmma benchmarks with `compile.sh`/`run.sh` per subdir.
- `TeBenchMark/`: transformer/linear/LLM performance scripts (Python, shell) and logs; see `TeBenchMark/README.md`.

## Build, Test, and Development Commands
- Regular units: `make -C RegularUnits` (outputs to `RegularUnits/bin`); individual targets: `make` inside each subfolder.
- DPX/TMA/DSM: `make -C NewFeatures/DPX` or per-subdir `make`; run `NewFeatures/DPX/run_all.sh` when provided.
- Tensor Cores: run the per-subdir scripts (e.g., `TensorCores/mma/run_all.sh`, `TensorCores/wgmma/throughput/run.sh`).
- CoWoS: `make -C NewFeatures/CoWoS`，再 `NewFeatures/CoWoS/run_all.sh [compute_dev] [mem_dev]`（默认单卡 `0 0`；两卡 P2P 仅用于调试）。
- tcgen05（B200 第五代 Tensor Core/tmem）：`make -C TensorCores/tcgen05`，`TensorCores/tcgen05/run.sh {ldst|cp_shift|mma_ws}`（需 SM100+）。
- TeBenchMark: follow `TeBenchMark/README.md` for Docker launch, then `python ./linear/linear.py` or `bash ./models/llama.sh`.

## Coding Style & Naming Conventions
- CUDA/C++: follow existing 2-space/tab alignment; descriptive kernel names (e.g., `tma_bw_1d`, `vimax3_s32_lat`).
- Python: PEP 8, snake_case for functions/files, PascalCase for classes.
- Mirror measured primitive/topology in filenames; keep folder naming consistent with existing latency/bandwidth patterns.
- Prefer `Common/` helpers over duplicating utilities.

## Testing Guidelines
- Build checks: `make` at repo root and inside any new benchmark folder.
- Functional runs: `./run.sh` or module-specific scripts on target GPUs; store outputs in `log/` when relevant.
- Naming: keep `*_lat`, `*_bw`, `*throughput` for comparability.
- Document assumptions (SM count, clocks, GPU model) via brief README updates or inline comments.

## Commit & Pull Request Guidelines
- Commits: short imperative summaries; scope prefix optional (history examples: “Update log (remove useless imformation)”, “init”).
- PRs: describe benchmark scope, target GPU/driver/CUDA version, and reproduction steps (commands + environment).
- Attach logs/plots from `log/` when modifying performance-sensitive kernels; note any non-default clock/power settings.
- Link related paper sections/issues when relevant.

## B200 适配重点
- 目标：将基准扩展到 Blackwell B200，覆盖新架构特性与延迟/吞吐变化。
- 编译：各 `Makefile` 已加入 B200 对应 `-gencode`（`sm_100`，以 PTX 文档为准），保持向后兼容。
- Tensor Core 第五代：根据 PTX 文档补充 B200 专属指令与数据类型路径，必要时新增 TensorCores kernel 与脚本。
- 封装差异：B200 为 2×B100 CoWoS，HBM 与缓存层级延迟可能不同；已添加 CoWoS 跨封装访存用例（默认单卡验证，P2P 模式仅供多卡调试），建议将实测结果输出到新的 B200 `log/` 目录。
- 文档：在 README/脚本注释标明 B200 依赖（CUDA/驱动版本、示例命令）并记录测试硬件。

## Security & Configuration Tips
- 仅在隔离的实验主机或容器中运行；避免提交凭证或私有数据。
- 确认 CUDA 版本与驱动满足目标 GPU（含 B200）要求；TeBenchMark 推荐使用文档中的 NVIDIA PyTorch 容器。
