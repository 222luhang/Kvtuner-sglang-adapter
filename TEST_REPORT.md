# KVTuner-SGLang 适配测试报告

## 项目信息
- **项目名称**: KVTuner-SGLang Adapter
- **GitHub 仓库**: https://github.com/luhang222/kvtuner-sglang-adapter
- **创建时间**: 2026-02-11
- **测试环境**: 
  - Machine 1: 10.60.252.76 (2x RTX 4090)
  - Machine 2: 10.60.61.227 (2x RTX 4090)

## 完成内容

### 1. 核心适配代码

#### kvtuner_quant.py (364行)
- **KVTunerQuantConfig**: 量化配置类
  - nbits_key/value: 2/4/8 bit 量化
  - asym: 对称/非对称量化
  - axis: 0=per-token, 1=per-channel
  - residual_length: 残差缓存长度

- **KVTunerVanillaQuantizer**: 量化器实现
  - 对称量化: `scale = max(|x|) / q_max`
  - 非对称量化: `scale = (max-min)/(q_max-q_min), zero = min/scale`

- **KVTunerQuantizedKVCache**: 量化缓存管理
  - 支持层-wise 混合精度
  - 残差缓存策略

#### kvtuner_kv_pool.py (248行)
- **KVTunerMHATokenToKVPool**: 扩展 SGLang MHATokenToKVPool
  - 重写 `set_kv_buffer()`: 添加量化逻辑
  - 重写 `get_kv_buffer()`: 添加反量化逻辑
  - 内存统计功能

### 2. 辅助脚本

| 脚本 | 功能 |
|------|------|
| `apply_kvtuner_patch.py` | 自动应用 SGLang 补丁 |
| `setup_kvtuner.sh` | 一键安装脚本 |
| `launch_distributed_kvtuner.sh` | 分布式启动脚本 |
| `test_kvtuner.py` | 功能测试脚本 |
| `demo_kvtuner.py` | 演示脚本 |

### 3. 启动命令

**Node 0 (10.60.252.76)**:
```bash
python -m sglang.launch_server \
  --model-path /data/Qwen/Qwen2.5-7B \
  --tp-size 4 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr "10.60.252.76:29500" \
  --enable-kvtuner-quant \
  --kvtuner-nbits-key 4 \
  --kvtuner-nbits-value 4
```

**Node 1 (10.60.61.227)**:
```bash
python -m sglang.launch_server \
  --model-path /data/Qwen/Qwen2.5-7B \
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
| `--enable-kvtuner-quant` | bool | False | 启用量化 |
| `--kvtuner-nbits-key` | int | 4 | Key 量化位数 |
| `--kvtuner-nbits-value` | int | 4 | Value 量化位数 |
| `--kvtuner-asym` | bool | False | 非对称量化 |
| `--kvtuner-axis-key` | int | 0 | Key 量化轴 |
| `--kvtuner-axis-value` | int | 0 | Value 量化轴 |
| `--kvtuner-q-group-size` | int | 64 | 量化组大小 |
| `--kvtuner-residual-length` | int | 128 | 残差长度 |

## 性能预期

| 配置 | 显存节省 | 精度损失 |
|------|----------|----------|
| FP16 (Baseline) | 0% | 0% |
| KV8 | ~50% | <0.5% |
| KV4 (推荐) | ~75% | <1% |
| KV2 | ~87.5% | ~2-3% |

## 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/luhang222/kvtuner-sglang-adapter.git
cd kvtuner-sglang-adapter

# 2. 安装 KVTuner
cd /path/to/kvtuner/flexible_quant
pip install -e .

# 3. 安装 SGLang
cd /path/to/sglang
pip install -e "python[all]"

# 4. 应用补丁
python apply_kvtuner_patch.py
```

## 测试计划

### 单节点测试 (2 GPU)
```bash
python -m sglang.launch_server \
  --model-path /data/Qwen/Qwen2.5-7B \
  --tp-size 2 \
  --enable-kvtuner-quant \
  --kvtuner-nbits-key 4 \
  --kvtuner-nbits-value 4
```

### 测试请求
```bash
curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "你好", "sampling_params": {"temperature": 0}}'
```

## 已知限制

1. SGLang 安装需要编译 CUDA 扩展，耗时约 10-15 分钟
2. GitHub 推送需要 Personal Access Token (PAT)
3. 双机通信需要确保 29500 端口开放

## 下一步工作

- [ ] 完成 SGLang 在两台机器上的安装
- [ ] 验证 KVTuner 补丁正确应用
- [ ] 使用 Qwen2.5-7B 进行单节点测试
- [ ] 进行双机 4 GPU 分布式测试
- [ ] 性能基准测试对比

## 参考资料

- KVTuner Paper: [ICML 2025]
- SGLang: https://github.com/sgl-project/sglang
- KVTuner: https://github.com/cmd2001/KVTuner

---
报告生成时间: 2026-02-11 20:33
