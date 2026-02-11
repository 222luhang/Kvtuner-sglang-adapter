# KVTuner 适配 SGLang 完整方案

## 1. 架构分析

### SGLang KV Cache 架构
- **存储层**: `MHATokenToKVPool` 类管理实际KV缓存存储
- **访问层**: `set_kv_buffer()` 存储KV, `get_kv_buffer()` 读取KV
- **注意力层**: `RadixAttention` 处理注意力计算

### KVTuner 核心组件
- **FlexibleQuantizedCache**: 量化缓存配置和管理
- **VanillaQuantizer**: 对称/非对称量化实现
- **配置参数**: nbits_key, nbits_value, axis_key, axis_value, q_group_size, residual_length

## 2. 需要修改的文件

### 2.1 新增文件
1. `python/sglang/srt/layers/quantization/kvtuner_quant.py` - KVTuner量化适配层
2. `python/sglang/srt/mem_cache/kvtuner_kv_pool.py` - 量化KV缓存池

### 2.2 修改文件
1. `python/sglang/srt/mem_cache/memory_pool.py` - 集成量化支持
2. `python/sglang/srt/layers/quantization/__init__.py` - 注册量化方法
3. `python/sglang/srt/server_args.py` - 添加KVTuner配置参数
4. `python/sglang/srt/mem_cache/allocator.py` - 支持量化缓存分配

## 3. 具体实现

### 3.1 KVTuner 量化配置
```python
# KVTuner 支持的配置
- nbits_key: 2, 4, 8 (key量化位数)
- nbits_value: 2, 4, 8 (value量化位数)
- asym: True/False (是否非对称量化)
- axis_key: 0 (per-token), 1 (per-channel)
- axis_value: 0 (per-token), 1 (per-channel)
- q_group_size: 64 (量化组大小)
- residual_length: 128 (保留的原始精度token数)
```

### 3.2 双机4 GPU分布式配置
```
机器1 (10.60.252.76): GPU 0, 1
机器2 (10.60.61.227): GPU 2, 3
TP_SIZE=4 (Tensor Parallel)
```

## 4. 启动命令

### 机器1 (10.60.252.76) - Node 0
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-2-7b-hf \
  --tp-size 4 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr "10.60.252.76:29500" \
  --enable-kvtuner-quant \
  --kvtuner-nbits-key 4 \
  --kvtuner-nbits-value 4 \
  --kvtuner-asym false \
  --kvtuner-axis-key 0 \
  --kvtuner-axis-value 0 \
  --kvtuner-q-group-size 64 \
  --kvtuner-residual-length 128
```

### 机器2 (10.60.61.227) - Node 1
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-2-7b-hf \
  --tp-size 4 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr "10.60.252.76:29500" \
  --enable-kvtuner-quant \
  --kvtuner-nbits-key 4 \
  --kvtuner-nbits-value 4 \
  --kvtuner-asym false \
  --kvtuner-axis-key 0 \
  --kvtuner-axis-value 0 \
  --kvtuner-q-group-size 64 \
  --kvtuner-residual-length 128
```
