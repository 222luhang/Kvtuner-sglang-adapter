# KVTuner-SGLang Integration

将 KVTuner 的 KV Cache 量化适配到 SGLang，支持双机 4 GPU 分布式推理。

## 已完成工作

### 1. 核心适配代码
- `python/sglang/srt/layers/quantization/kvtuner_quant.py` - KVTuner 量化适配层
- `python/sglang/srt/mem_cache/kvtuner_kv_pool.py` - 量化 KV 缓存池

### 2. 补丁脚本
- `apply_kvtuner_patch.py` - 自动应用 SGLang 修改补丁
- `setup_kvtuner.sh` - 一键安装脚本

### 3. 支持的功能
- KV Cache 量化（Key/Value 独立配置）
- 对称/非对称量化
- Per-token/Per-channel 量化轴选择
- 残差缓存（保持最新 token 全精度）
- 层-wise 混合精度支持

## 双机 4 GPU 配置

| 机器 | IP | GPU |
|------|-----|-----|
| Node 0 | 10.60.252.76 | GPU 0, 1 |
| Node 1 | 10.60.61.227 | GPU 2, 3 |

## 快速开始

### 1. 安装依赖
```bash
cd /home/ubuntu/.openclaw/workspace/sglang
./setup_kvtuner.sh
```

### 2. 启动分布式服务

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

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-kvtuner-quant` | False | 启用 KVTuner 量化 |
| `--kvtuner-nbits-key` | 4 | Key 量化位数 (2/4/8) |
| `--kvtuner-nbits-value` | 4 | Value 量化位数 (2/4/8) |
| `--kvtuner-asym` | False | 使用非对称量化 |
| `--kvtuner-axis-key` | 0 | Key 量化轴 (0=per-token, 1=per-channel) |
| `--kvtuner-axis-value` | 0 | Value 量化轴 (0=per-token, 1=per-channel) |
| `--kvtuner-q-group-size` | 64 | 量化组大小 |
| `--kvtuner-residual-length` | 128 | 残差缓存长度 |

## 文件说明

- `KVTUNER_ADAPTATION_PLAN.md` - 详细适配方案
- `kvtuner_quant.py` - 核心量化实现
- `kvtuner_kv_pool.py` - KV 缓存池扩展
- `apply_kvtuner_patch.py` - SGLang 补丁脚本

## 进度状态

- [x] 代码架构分析
- [x] 核心适配代码实现
- [x] 补丁脚本创建
- [ ] SGLang 安装验证
- [ ] 功能测试
- [ ] 性能基准测试
