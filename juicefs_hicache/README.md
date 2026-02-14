# JuiceFS HiCache Storage Backend for SGLang

## 项目简介

本项目实现了一个基于 JuiceFS 的 SGLang HiCache L3 存储后端，使得 SGLang 可以将 KV Cache 存储到 JuiceFS 分布式文件系统中。

## 核心特性

- ✅ 实现 HiCacheStorage 抽象类的三个核心方法：`get()`, `set()`, `exists()`
- ✅ 支持批量操作：`batch_get()`, `batch_set()`, `batch_exists()`
- ✅ 零拷贝数据传输（Zero-copy）
- ✅ 分片存储（256 shards）提高并发性能
- ✅ 完整的统计信息
- ✅ 支持动态加载

## 文件说明

| 文件 | 说明 |
|------|------|
| `juicefs_hicache_storage.py` | 核心存储后端实现 |
| `register_juicefs_backend.py` | 后端注册脚本 |
| `test_juicefs_backend.py` | 功能测试脚本 |
| `launch_sglang_juicefs.sh` | 启动脚本示例 |
| `README.md` | 详细使用文档 |

## 快速开始

### 1. 环境准备

```bash
# 安装 JuiceFS
wget https://github.com/juicedata/juicefs/releases/download/v1.1.0/juicefs-1.1.0-linux-amd64.tar.gz
tar -xzf juicefs-1.1.0-linux-amd64.tar.gz
sudo mv juicefs /usr/local/bin/

# 格式化并挂载 JuiceFS
juicefs format --storage file --bucket /dev/vdc/jfs-data redis://localhost:6379/1 myjfs
juicefs mount --cache-dir /dev/vdc/jfs-cache redis://localhost:6379/1 /mnt/jfs
```

### 2. 注册后端

```bash
python register_juicefs_backend.py
```

### 3. 启动 SGLang

```bash
python -m sglang.launch_server \
    --model-path /data/Qwen/Qwen2.5-7B \
    --enable-hierarchical-cache \
    --hicache-storage-backend juicefs \
    --hicache-storage-backend-config '{
        "mount_point": "/mnt/jfs",
        "use_direct_io": false
    }'
```

## 核心方法实现

### 1. `get(key, target_location)` - 读取 KV Cache

```python
def get(self, key, target_location):
    file_path = self._get_file_path(key)
    
    with open(file_path, 'rb') as f:
        buf = target_location.view(torch.uint8).contiguous().numpy()
        f.readinto(buf)  # 零拷贝读取
    
    return target_location
```

### 2. `set(key, target_location)` - 写入 KV Cache

```python
def set(self, key, target_location):
    file_path = self._get_file_path(key)
    data_tensor = target_location.contiguous().cpu()
    data_bytes = data_tensor.view(torch.uint8).numpy().tobytes()
    
    with open(file_path, 'wb') as f:
        f.write(data_bytes)
    
    return True
```

### 3. `exists(key)` - 检查 Key 是否存在

```python
def exists(self, key):
    return self._get_file_path(key).exists()
```

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    SGLang with HiCache                       │
├─────────────────────────────────────────────────────────────┤
│  L1: GPU KV Cache                                            │
│       ↓ eviction                                             │
│  L2: Host Memory                                             │
│       ↓ write-back                                           │
│  L3: JuiceFS Storage ←──┐                                    │
│       │                  │                                    │
│       │  ┌───────────────┼─────────────────────┐             │
│       │  │               │                     │             │
│       └──►  /mnt/jfs     │  JuiceFS Mount      │             │
│              ├── hicache/│                     │             │
│              │   ├── shard_00/                 │             │
│              │   ├── shard_01/                 │             │
│              │   └── ...                       │             │
│              │                                 │             │
│              └──► /dev/vdc/jfs-cache           │             │
│                      (Local Cache)             │             │
└─────────────────────────────────────────────────────────────┘
```

## 测试验证

```bash
$ python test_juicefs_backend.py

============================================================
JuiceFS HiCache Backend Test Suite
============================================================

============================================================
Testing JuiceFS HiCache Backend - Basic Operations
============================================================
✅ Backend created successfully
✅ Set operation successful: test_key_001
✅ Exists check passed: test_key_001 exists
✅ Get operation successful: data verified
Backend stats:
  - Get count: 1
  - Set count: 1
  - Hit count: 1
  - Hit rate: 100.00%

============================================================
✅ All basic tests passed!

============================================================
TEST SUMMARY
============================================================
  ✅ PASSED: Basic Operations
  ✅ PASSED: Batch Operations
  ✅ PASSED: Factory Function

============================================================
✅ ALL TESTS PASSED!
```

## 配置选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mount_point` | str | "/mnt/jfs" | JuiceFS 挂载点 |
| `use_direct_io` | bool | false | 是否使用 Direct I/O |
| `enable_compression` | bool | false | 是否启用压缩 |

## 许可证

Apache-2.0

## 参考

- [SGLang HiCache 文档](https://docs.sglang.com.cn/advanced_features/hicache_design.html)
- [JuiceFS 文档](https://juicefs.com/docs/community/introduction/)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
