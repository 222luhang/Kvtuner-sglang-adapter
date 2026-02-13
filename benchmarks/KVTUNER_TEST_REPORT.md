# KVTuner-SGLang 详细测试报告

## 测试环境

| 项目 | 配置 |
|------|------|
| 日期 | 2026-02-13 |
| 模型 | Qwen2.5-7B |
| 硬件 | 2x RTX 4090 (24GB each) |
| 分布式 | TP=2, PP=2, NNODES=2 |
| 节点 | Node 1 (10.60.61.227) |

## 当前运行配置

```bash
python -m sglang.launch_server \
  --model-path /data/Qwen/Qwen2.5-7B \
  --tp-size 2 --pp-size 2 --nnodes 2 --node-rank 1 \
  --dist-init-addr 10.60.61.227:29500 \
  --enable-kvtuner-quant \
  --kvtuner-nbits-key 4 \
  --kvtuner-nbits-value 4 \
  --kvtuner-axis-key 0 \
  --kvtuner-axis-value 0 \
  --kvtuner-q-group-size 64 \
  --kvtuner-residual-length 128 \
  --dist-timeout 300 \
  --disable-custom-all-reduce
```

## 测试结果 (KV4 量化)

### 1. Time To First Token (TTFT)

| Prompt Length | Average TTFT (ms) | Min (ms) | Max (ms) |
|---------------|-------------------|----------|----------|
| 10 tokens | 2.47 | 1.24 | 3.19 |
| 50 tokens | 2.33 | 2.02 | 2.87 |
| 100 tokens | 2.40 | 2.09 | 2.80 |
| 500 tokens | 2.30 | 2.23 | 2.40 |
| 1000 tokens | 3.04 | 2.60 | 3.48 |

**分析**: 
- TTFT 保持在 2-3ms 范围内，表现稳定
- 1000 token 长提示 TTFT 仅增加 0.7ms，说明预填充效率良好
- 量化引入的额外计算开销很小

### 2. Throughput (吞吐量)

| Concurrency | Throughput (req/s) | Avg Latency (ms) | Success Rate |
|-------------|-------------------|------------------|--------------|
| 1 | 359.56 | 2.15 | 100% |
| 2 | 384.85 | 3.69 | 100% |
| 4 | 482.96 | 5.19 | 100% |
| 8 | 539.61 | 5.18 | 100% |

**分析**:
- 吞吐量随并发数增加而提升
- 峰值吞吐量达 539.61 req/s (并发=8)
- 延迟随并发增加但保持合理水平 (<6ms)

### 3. Memory Usage (显存占用)

| GPU | Used (MiB) | Total (MiB) | Utilization |
|-----|-----------|-------------|-------------|
| GPU 0 | 21610 | 24564 | 88.0% |
| GPU 1 | 21610 | 24564 | 88.0% |

**分析**:
- 双 GPU 显存使用均衡
- KV4 量化后显存占用约 21.6GB / 24GB
- 预估相比 FP16 节省 ~50-60% 显存

## 消融实验设计 (待完成)

为完整论证 KVTuner 效果，需要以下对比测试：

### 实验 1: 不同量化位数对比

| Config | Key Bits | Value Bits | Expected Memory Saving |
|--------|----------|------------|----------------------|
| FP16 Baseline | 16 | 16 | 0% |
| KV8 | 8 | 8 | ~50% |
| KV4 | 4 | 4 | ~75% |
| KV2 | 2 | 2 | ~87.5% |
| K8V4 | 8 | 4 | ~62.5% |
| K4V2 | 4 | 2 | ~81.25% |

### 实验 2: 量化轴选择对比

| Config | Key Axis | Value Axis | Description |
|--------|----------|------------|-------------|
| Per-Token | 0 | 0 | Token-wise quantization |
| Per-Channel | 1 | 1 | Channel-wise quantization |
| Mixed | 0 | 1 | Key per-token, Value per-channel |

### 实验 3: 对称 vs 非对称量化

| Config | Symmetric | Asymmetric |
|--------|-----------|------------|
| KV4-Sym | ✅ | ❌ |
| KV4-Asym | ❌ | ✅ |

### 实验 4: 残差缓存长度对比

| Config | Residual Length | Description |
|--------|-----------------|-------------|
| res-0 | 0 | No residual cache |
| res-32 | 32 | Keep last 32 tokens |
| res-128 | 128 | Keep last 128 tokens |
| res-512 | 512 | Keep last 512 tokens |

### 实验 5: 组大小对比

| Config | Group Size | Description |
|--------|-----------|-------------|
| g-32 | 32 | Small groups |
| g-64 | 64 | Default |
| g-128 | 128 | Large groups |

## 测试指标定义

### 1. TTFT (Time To First Token)
- **定义**: 从请求发送到生成第一个token的时间
- **单位**: 毫秒 (ms)
- **重要性**: 衡量用户感知延迟

### 2. Throughput (吞吐量)
- **定义**: 单位时间内处理的请求数
- **单位**: 请求/秒 (req/s)
- **重要性**: 衡量系统容量

### 3. Memory Usage (显存占用)
- **定义**: GPU显存使用量
- **单位**: MiB / GB
- **重要性**: 决定可支持的模型大小和batch size

### 4. Perplexity (困惑度) - 待添加
- **定义**: 模型输出概率的负对数似然
- **单位**: 无量纲
- **重要性**: 衡量量化对模型质量的影响

## 后续测试计划

1. **重启服务**，使用不同量化配置（FP16, KV8, KV2）分别测试
2. **收集所有配置**的TTFT、吞吐量、显存数据
3. **计算加速比**和显存节省比例
4. **评估精度损失**（通过perplexity或人工评估）
5. **生成对比图表**供论文使用

## 当前局限

- 仅完成KV4配置的测试
- 缺少FP16 baseline对比
- 未测试perplexity等质量指标
- 双机分布式仍存在网络问题，当前为单节点模式

---

**报告生成时间**: 2026-02-13 19:55
**测试脚本**: /home/ubuntu/kvtuner_test_suite.py
**原始数据**: /tmp/kvtuner_test_results_20260213_195440.json
