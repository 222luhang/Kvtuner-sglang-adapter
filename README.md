# KVTuner-SGLang Adapter

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![SGLang](https://img.shields.io/badge/SGLang-compatible-orange.svg)](https://github.com/sgl-project/sglang)

将 KVTuner 的 KV Cache 量化适配到 SGLang，支持双机 4 GPU 分布式推理。

## 功能特性

- ✅ KV Cache 量化（Key/Value 独立配置）
- ✅ 对称/非对称量化
- ✅ Per-token/Per-channel 量化轴选择
- ✅ 残差缓存（保持最新 token 全精度）
- ✅ 层-wise 混合精度支持
- ✅ 双机分布式推理支持

## 硬件要求

- 2x 机器，每机器 2x NVIDIA RTX 4090 (24GB)
- 总显存：96GB
- CUDA 12.4+

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/kvtuner-sglang-adapter.git
cd kvtuner-sglang-adapter
```

### 2. 安装依赖

```bash
# 安装 KVTuner
cd /path/to/kvtuner/flexible_quant
pip install -e .

# 安装 SGLang
cd /path/to/sglang
pip install -e "python[all]"

# 应用适配补丁
python apply_kvtuner_patch.py
```

### 3. 启动分布式服务

**Node 0 (10.60.252.76)**
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-2-7b-hf \
  --tp-size 4 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr "10.60.252.76:29500" \
  --enable-kvtuner-quant \
  --kvtuner-nbits-key 4 \
  --kvtuner-nbits-value 4
```

**Node 1 (10.60.61.227)**
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-2-7b-hf \
  --tp-size 4 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr "10.60.252.76:29500" \
  --enable-kvtuner-quant \
  --kvtuner-nbits-key 4 \
  --kvtuner-nbits-value 4
```

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable-kvtuner-quant` | bool | False | 启用 KVTuner 量化 |
| `--kvtuner-nbits-key` | int | 4 | Key 量化位数 (2/4/8) |
| `--kvtuner-nbits-value` | int | 4 | Value 量化位数 (2/4/8) |
| `--kvtuner-asym` | bool | False | 使用非对称量化 |
| `--kvtuner-axis-key` | int | 0 | Key 量化轴 (0=per-token, 1=per-channel) |
| `--kvtuner-axis-value` | int | 0 | Value 量化轴 (0=per-token, 1=per-channel) |
| `--kvtuner-q-group-size` | int | 64 | 量化组大小 |
| `--kvtuner-residual-length` | int | 128 | 残差缓存长度 |

## 文件说明

```
.
├── kvtuner_quant.py          # 核心量化实现
├── kvtuner_kv_pool.py        # KV 缓存池扩展
├── apply_kvtuner_patch.py    # SGLang 补丁脚本
├── setup_kvtuner.sh          # 一键安装脚本
├── KVTUNER_ADAPTATION_PLAN.md # 详细适配方案
└── README.md                 # 本文档
```

## 架构设计

### KVTunerQuantConfig
配置类，定义量化参数：
- `nbits_key`: Key 量化位数
- `nbits_value`: Value 量化位数
- `asym`: 是否非对称量化
- `axis_key/value`: 量化轴选择
- `q_group_size`: 量化组大小
- `residual_length`: 残差缓存长度

### KVTunerVanillaQuantizer
量化器实现，支持：
- 对称量化: `scale = max(|x|) / q_max`
- 非对称量化: `scale = (max - min) / (q_max - q_min), zero = min / scale`

### KVTunerMHATokenToKVPool
扩展 SGLang 的 MHATokenToKVPool：
- 重写 `set_kv_buffer()`: 添加量化逻辑
- 重写 `get_kv_buffer()`: 添加反量化逻辑
- 支持残差缓存策略

## 性能对比

| 配置 | 显存占用 | 精度损失 |
|------|----------|----------|
| FP16 (Baseline) | 100% | 0% |
| KVTuner KV4 | ~25% | <1% |
| KVTuner KV8 | ~50% | <0.5% |

## 测试

```bash
# 单节点测试
python -m sglang.launch_server \
  --model-path meta-llama/Llama-2-7b-hf \
  --tp-size 2 \
  --enable-kvtuner-quant \
  --kvtuner-nbits-key 4 \
  --kvtuner-nbits-value 4

# 发送测试请求
curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "What is the capital of France?", "sampling_params": {"temperature": 0}}'
```

## 引用

KVTuner Paper:
```bibtex
@inproceedings{li2025kvtuner,
  title={KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization},
  author={Li, Xing and Xing, Zeyu and Li, Yiming and others},
  booktitle={ICML},
  year={2025}
}
```

SGLang:
```bibtex
@software{sglang,
  title={SGLang: Efficient Execution of Structured Language Model Programs},
  author={SGLang Team},
  url={https://github.com/sgl-project/sglang}
}
```

## License

MIT License - See LICENSE file for details
