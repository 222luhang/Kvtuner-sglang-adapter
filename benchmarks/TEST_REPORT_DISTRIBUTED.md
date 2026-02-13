# KVTuner-SGLang 分布式测试报告 (2026-02-13)

## 测试配置

| 项目 | 配置 |
|------|------|
| 日期 | 2026-02-13 22:57 |
| 模型 | Qwen2.5-7B |
| 分布式 | TP=2, PP=2, NNODES=2 |
| 量化 | KV4 (Key=4bit, Value=4bit) |
| Node 0 | 10.60.61.227 (Master) |
| Node 1 | 10.60.252.76 (Worker) |

## 测试结果

### 1. Time To First Token (TTFT)

| Prompt Length | Average (ms) | Min (ms) | Max (ms) |
|---------------|--------------|----------|----------|
| 10 tokens | 2.55 | 1.35 | 3.78 |
| 50 tokens | 2.62 | 1.69 | 3.82 |
| 100 tokens | 2.89 | 2.60 | 3.16 |
| 500 tokens | 2.33 | 2.04 | 2.68 |
| 1000 tokens | 2.08 | 1.89 | 2.23 |

**分析**: TTFT 保持在 2-3ms，表现稳定。长 prompt (1000 tokens) 的 TTFT 反而更低，可能是缓存预热效应。

### 2. Throughput (吞吐量)

| Concurrency | Throughput (req/s) | Avg Latency (ms) | Success Rate |
|-------------|-------------------|------------------|--------------|
| 1 | 332.75 | 2.48 | 100% |
| 2 | 543.94 | 2.59 | 100% |
| 4 | 583.60 | 4.07 | 100% |
| 8 | 570.29 | 5.01 | 100% |

**分析**: 峰值吞吐量达 **583.60 req/s** (并发=4)，成功率 100%。

### 3. Memory Usage (显存占用)

| GPU | Used (MiB) | Total (MiB) | Utilization |
|-----|-----------|-------------|-------------|
| GPU 0 | 21610 | 24564 | 88.0% |
| GPU 1 | 21610 | 24564 | 88.0% |

**分析**: 双 GPU 显存使用均衡，KV4 量化后占用约 21.6GB/24GB。

## 分布式架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Distributed SGLang Cluster                │
├─────────────────────────┬───────────────────────────────────┤
│      Node 0 (Master)    │         Node 1 (Worker)           │
│    10.60.61.227:30000   │      10.60.252.76:30001           │
│  ┌─────┐    ┌─────┐     │    ┌─────┐    ┌─────┐             │
│  │GPU 0│    │GPU 1│     │    │GPU 2│    │GPU 3│             │
│  │ PP0 │    │ PP0 │     │    │ PP1 │    │ PP1 │             │
│  │ TP0 │    │ TP1 │     │    │ TP0 │    │ TP1 │             │
│  └─────┘    └─────┘     │    └─────┘    └─────┘             │
│        TP=2 (Tensor)    │         TP=2 (Tensor)               │
│        PP=0 (Pipeline)  │         PP=1 (Pipeline)             │
├─────────────────────────┴───────────────────────────────────┤
│                    NCCL Cross-Node Communication             │
└─────────────────────────────────────────────────────────────┘
```

## 关键配置

```bash
# Node 0 (Master)
python -m sglang.launch_server \
  --model-path /data/Qwen/Qwen2.5-7B \
  --tp-size 2 --pp-size 2 --nnodes 2 --node-rank 0 \
  --dist-init-addr 10.60.61.227:29500 \
  --enable-kvtuner-quant \
  --kvtuner-nbits-key 4 --kvtuner-nbits-value 4 \
  --dist-timeout 300 --disable-custom-all-reduce

# Node 1 (Worker)
python -m sglang.launch_server \
  --model-path /data/Qwen/Qwen2.5-7B \
  --tp-size 2 --pp-size 2 --nnodes 2 --node-rank 1 \
  --dist-init-addr 10.60.61.227:29500 \
  --enable-kvtuner-quant \
  --kvtuner-nbits-key 4 --kvtuner-nbits-value 4 \
  --dist-timeout 300 --disable-custom-all-reduce
```

## 后续测试计划

为完成论文消融实验，建议进行以下对比测试：

| 配置 | 参数 |
|------|------|
| FP16 Baseline | `--enable-kvtuner-quant=False` |
| KV8 | `--kvtuner-nbits-key 8 --kvtuner-nbits-value 8` |
| KV4 (当前) | `--kvtuner-nbits-key 4 --kvtuner-nbits-value 4` ✅ |
| KV2 | `--kvtuner-nbits-key 2 --kvtuner-nbits-value 2` |

---

**测试时间**: 2026-02-13 22:57
**GitHub**: https://github.com/222luhang/kvtuner-sglang-adapter
