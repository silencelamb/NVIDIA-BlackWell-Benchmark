# Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis

This repository contains the code for benchmarking NVIDIA GPU performance. The relevant papers are as follows:

- Weile Luo, Ruibo Fan, Zeyu Li, Dayou Du, Qiang Wang, and Xiaowen Chu. "[Benchmarking and Dissecting the Nvidia Hopper GPU Architecture.](https://ieeexplore.ieee.org/abstract/document/10579250?casa_token=tw5ix8vdZvsAAAAA:9K3snl2qTP8Pf_ZIN-3T9RCil_PniO2LVrRMPxP5gr8eUYdnag9L_YkhYsFdydmXtYQWfuz47ZKXQ4o)" In 2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS), pp. 656-667. IEEE, 2024.
- Weile Luo, Ruibo Fan, Zeyu Li, Dayou Du, Hongyuan Liu, Qiang Wang, and Xiaowen Chu. "[Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis.](https://arxiv.org/abs/2501.12084)" arXiv preprint arXiv:2501.12084 (2025).

If you find this work useful, please cite this project and our papers.

```
@inproceedings{luo2024benchmarking,
  title={Benchmarking and dissecting the nvidia hopper gpu architecture},
  author={Luo, Weile and Fan, Ruibo and Li, Zeyu and Du, Dayou and Wang, Qiang and Chu, Xiaowen},
  booktitle={2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
  pages={656--667},
  year={2024},
  organization={IEEE}
}

@article{luo2025dissecting,
  title={Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis},
  author={Luo, Weile and Fan, Ruibo and Li, Zeyu and Du, Dayou and Liu, Hongyuan and Wang, Qiang and Chu, Xiaowen},
  journal={arXiv preprint arXiv:2501.12084},
  year={2025}
}
```

## Recommended environment

- CUDA 12.6 or above (update to the B200-supported CUDA/driver when targeting Blackwell)
- Ubuntu 20.04

## Build & Usage

- Regular units: `make -C RegularUnits` (produces `RegularUnits/bin`).
- DPX/TMA/DSM: `make -C NewFeatures/DPX` or per-subdir `make`; run scripts such as `NewFeatures/DPX/run_all.sh`.
- Tensor Cores: per-directory scripts such as `TensorCores/mma/run_all.sh` or `TensorCores/wgmma/*/run.sh`.
- tcgen05（B200 第五代 Tensor Core/tmem）：`make -C TensorCores/tcgen05`，按需运行 `TensorCores/tcgen05/run.sh {ldst|cp_shift|mma_ws}`（默认 f16，需 SM100+）。
- TeBenchMark: follow `TeBenchMark/README.md` (Docker-based) and run `python ./linear/linear.py` or `bash ./models/llama.sh`.
- CoWoS（B200 跨封装访存）: `make -C NewFeatures/CoWoS`，再运行 `NewFeatures/CoWoS/run_all.sh [compute_dev] [mem_dev]`（默认单卡 `0 0`；如有两卡且支持 P2P，可指定不同设备）。

## B200 Notes

- 已在各 Makefile 中加入 `sm_100` gencode 以覆盖 Blackwell B200；如需精确指令路径，请以 PTX 文档为准。
- B200 的 Tensor Core 属于第五代，具有专属指令；相关 kernel/script 需按 PTX 文档补充或替换。
- B200 封装为 2×B100 CoWoS，HBM/缓存层级的延迟与带宽可能与 Hopper 不同；建议增加跨封装访存延迟/带宽用例并记录到新的日志目录。
- 当前 CoWoS 基准默认单卡运行（compute_dev==mem_dev），仅用于流程验证；多卡 P2P 模式仅用于调试，无法等价于 B200 封装内路径。
- tcgen05 测试仅在支持 SM100 的 GPU 上可编译运行；形状/描述符需依据 PTX 规范调整以覆盖更多数据类型与布局。

## Acknowledgment

- https://github.com/shen203/GPU_Microbenchmark provides a reference for our regular unit tests.
- https://github.com/RRZE-HPC/gpu-benches provides a reference for our memory and TMA random access tests.
- We used the tools in https://github.com/blackjack2015/NV-DVFS-Benchmark to test the power consumption.
