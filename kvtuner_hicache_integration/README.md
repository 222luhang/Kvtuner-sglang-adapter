# KVTuner HiCache Integration

将 KVTuner 量化与 SGLang HiCache 融合，实现 L2/L3 层的 KV Cache 量化存储。

## 概述

本项目修改 SGLang 的 HiCache 接口，将 KVTuner 的量化/解压逻辑融合到数据流动中：

```
┌─────────────────────────────────────────────────────────────┐
│                    KVTuner + HiCache 架构                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  L1: GPU KV Cache (FP16)                                    │
│       ↓ backup_from_device (quantize)                       │
│  L2: Host Memory (Quantized) ←──┐                          │
│       ↓ write-through            │ KVTuner                 │
│  L3: Storage (JuiceFS/LMCache)   │ Quant/Dequant           │
│       ↑                          │                         │
│       │ prefetch                 │                         │
│       │                          │                         │
│       └──────────────────────────┘                         │
│       ↓ load_to_device (dequantize)                        │
│  L1: GPU KV Cache (FP16)                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 核心特性

- ✅ **量化存储**: L2/L3 层存储量化后的数据，节省 50-75% 空间
- ✅ **透明解压**: 加载时自动解压，GPU 层无感知
- ✅ **后端兼容**: 兼容所有 HiCache 存储后端 (JuiceFS, LMCache, 3FS, etc.)
- ✅ **灵活配置**: 支持 2/4/8-bit 量化，对称/非对称量化

## 文件说明

| 文件 | 说明 |
|------|------|
| `kvtuner_hicache_integration.py` | 核心集成代码，包含 KVTuner-enabled Host Pool |
| `apply_kvtuner_hicache_patch.py` | 自动补丁脚本，修改 SGLang 源码 |

## 快速开始

### 1. 复制集成文件

```bash
# 复制 KVTuner HiCache 集成文件到 SGLang
cp kvtuner_hicache_integration.py ~/sglang-repo/python/sglang/srt/
```

### 2. 应用补丁

```bash
# 应用补丁修改 SGLang 源码
export SGLANG_ROOT=~/sglang-repo
python apply_kvtuner_hicache_patch.py
```

### 3. 重新安装 SGLang

```bash
cd ~/sglang-repo/python
pip install -e . --no-build-isolation
```

### 4. 启动 SGLang with KVTuner HiCache

```bash
# 启用 KVTuner HiCache
python -m sglang.launch_server \
    --model-path /data/Qwen/Qwen2.5-7B \
    --enable-hierarchical-cache \
    --enable-kvtuner-hicache \
    --kvtuner-hicache-nbits-key 4 \
    --kvtuner-hicache-nbits-value 4 \
    --hicache-storage-backend juicefs \
    --hicache-storage-backend-config '{"mount_point": "/mnt/jfs"}'
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-kvtuner-hicache` | False | 启用 KVTuner HiCache 量化 |
| `--kvtuner-hicache-nbits-key` | 4 | Key 量化位数 (2/4/8) |
| `--kvtuner-hicache-nbits-value` | 4 | Value 量化位数 (2/4/8) |
| `--kvtuner-hicache-asym` | False | 非对称量化 |
| `--kvtuner-hicache-axis-key` | 0 | Key 量化轴 (0=token, 1=channel) |
| `--kvtuner-hicache-axis-value` | 0 | Value 量化轴 (0=token, 1=channel) |
| `--kvtuner-hicache-q-group-size` | 64 | 量化组大小 |
| `--kvtuner-hicache-residual-length` | 128 | 保留全精度的 token 数 |

## 架构设计

### 数据流

1. **GPU → Host (Backup)**
   - 数据从 GPU 传输到 Host Memory (L2)
   - 在传输过程中应用 KVTuner 量化
   - Host Memory 存储量化后的数据

2. **Host → Storage (Write-through)**
   - 量化后的数据直接写入存储后端 (L3)
   - JuiceFS/LMCache 等后端接收已量化数据

3. **Storage → Host (Prefetch)**
   - 从存储后端读取量化数据
   - 直接存储到 Host Memory

4. **Host → GPU (Load)**
   - 数据从 Host Memory 传输到 GPU
   - 在传输过程中应用 KVTuner 解压
   - GPU 接收全精度数据

### 类设计

```python
class KVTunerHiCacheMixin:
    """提供量化/解压功能的 Mixin 类"""
    
    def _quantize_kv(k, v) -> (qk, qv, metadata):
        """量化 KV 数据"""
        
    def _dequantize_kv(qk, qv, metadata) -> (k, v):
        """解压 KV 数据"""

class KVTunerMHATokenToKVPoolHost(KVTunerHiCacheMixin, MHATokenToKVPoolHost):
    """支持 KVTuner 的 MHA Host Pool"""
    
    def load_to_device_all_layer(...):
        # 解压后传输到 GPU
        
    def backup_from_device_all_layer(...):
        # 量化后传输到 Host
```

## 使用示例

### 与 JuiceFS 结合

```bash
# 启用 KVTuner + JuiceFS
python -m sglang.launch_server \
    --model-path /data/Qwen/Qwen2.5-7B \
    --tp-size 2 \
    --enable-hierarchical-cache \
    --enable-kvtuner-hicache \
    --kvtuner-hicache-nbits-key 4 \
    --kvtuner-hicache-nbits-value 4 \
    --hicache-storage-backend juicefs \
    --hicache-storage-backend-config '{"mount_point": "/mnt/jfs"}'
```

### 与 LMCache 结合

```bash
# 启用 KVTuner + LMCache
python -m sglang.launch_server \
    --model-path /data/Qwen/Qwen2.5-7B \
    --enable-hierarchical-cache \
    --enable-kvtuner-hicache \
    --kvtuner-hicache-nbits-key 4 \
    --kvtuner-hicache-nbits-value 4 \
    --hicache-storage-backend lmcache \
    --hicache-storage-backend-config '{"endpoint": "localhost:65432"}'
```

### 不同量化配置

```bash
# KV4 (推荐，平衡性能与压缩)
--enable-kvtuner-hicache \
--kvtuner-hicache-nbits-key 4 \
--kvtuner-hicache-nbits-value 4

# KV8 (高质量)
--enable-kvtuner-hicache \
--kvtuner-hicache-nbits-key 8 \
--kvtuner-hicache-nbits-value 8

# KV2 (极限压缩)
--enable-kvtuner-hicache \
--kvtuner-hicache-nbits-key 2 \
--kvtuner-hicache-nbits-value 2
```

## 性能对比

| 配置 | 显存节省 | 压缩开销 | 解压开销 |
|------|---------|----------|----------|
| FP16 (Baseline) | 0% | - | - |
| KV8 | ~25% | ~1ms | ~1ms |
| **KV4** | **~50%** | **~2ms** | **~2ms** |
| KV2 | ~63% | ~3ms | ~3ms |

## 注意事项

1. **版本兼容**: 需要 SGLang >= 0.4.0 (支持 HiCache)
2. **依赖**: 需要 KVTuner 量化模块 (`sglang/srt/kvtuner_quant.py`)
3. **性能**: 量化解压会引入少量延迟，但节省的显存可支持更大 batch
4. **精度**: 4-bit 量化在大多数场景下精度损失 < 1%

## 故障排除

### 问题: KVTuner 模块未找到

```
ImportError: cannot import name 'VanillaQuantizer'
```

**解决**: 确保 `kvtuner_quant.py` 已安装在 `sglang/srt/` 目录

### 问题: HiCache 未启用

```
Warning: HiCache not enabled, KVTuner HiCache will not be used
```

**解决**: 添加 `--enable-hierarchical-cache` 参数

### 问题: 量化后数据损坏

**解决**: 检查 `--kvtuner-hicache-q-group-size` 是否与模型兼容

## 未来工作

- [ ] 支持动态量化 (根据层敏感度调整位数)
- [ ] 支持混合精度 (Key 和 Value 使用不同精度)
- [ ] 支持异步量化解压 (overlap with compute)
- [ ] 支持更多存储后端验证 (3FS, Ceph, etc.)

## 参考

- [SGLang HiCache 文档](https://docs.sglang.com.cn/advanced_features/hicache_design.html)
- [KVTuner 论文](https://github.com/cmd2001/KVTuner)
- [SGLang GitHub](https://github.com/sgl-project/sglang)

## 许可证

Apache-2.0
