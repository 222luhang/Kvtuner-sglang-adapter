# JuiceFS HiCache Storage Backend Configuration

## Overview

This module provides a JuiceFS-based storage backend for SGLang HiCache L3 layer,
enabling distributed KV cache storage with high-performance local caching.

## Prerequisites

### 1. Install JuiceFS

```bash
# Download JuiceFS
wget https://github.com/juicedata/juicefs/releases/download/v1.1.0/juicefs-1.1.0-linux-amd64.tar.gz
tar -xzf juicefs-1.1.0-linux-amd64.tar.gz
sudo mv juicefs /usr/local/bin/

# Verify installation
juicefs --version
```

### 2. Format and Mount JuiceFS

```bash
# Format JuiceFS (using Redis as metadata engine and local disk as storage)
# For production, use object storage (S3, OSS, etc.) as backend
juicefs format \
    --storage file \
    --bucket /dev/vdc/jfs-data \
    redis://localhost:6379/1 \
    myjfs

# Mount JuiceFS
mkdir -p /mnt/jfs
juicefs mount \
    --cache-dir /dev/vdc/jfs-cache \
    --cache-size 50000 \
    redis://localhost:6379/1 \
    /mnt/jfs

# Verify mount
df -h /mnt/jfs
```

### 3. Install Python Dependencies

```bash
pip install torch numpy
```

## Installation

### Method 1: Direct Registration (Recommended)

```bash
# Copy the backend file to your SGLang installation
cp juicefs_hicache_storage.py /path/to/sglang/python/sglang/srt/mem_cache/storage/
cp register_juicefs_backend.py /path/to/sglang/python/sglang/srt/mem_cache/storage/

# Register the backend
python /path/to/sglang/python/sglang/srt/mem_cache/storage/register_juicefs_backend.py
```

### Method 2: Dynamic Loading

SGLang supports dynamic backend loading without modifying core code:

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --hicache-storage-backend dynamic \
    --hicache-storage-backend-config '{
        "backend_name": "juicefs",
        "module_path": "juicefs_hicache_storage",
        "class_name": "HiCacheJuiceFS",
        "mount_point": "/mnt/jfs",
        "use_direct_io": false
    }'
```

## Usage

### Basic Usage

```bash
python -m sglang.launch_server \
    --model-path /data/Qwen/Qwen2.5-7B \
    --tp-size 2 \
    --enable-hierarchical-cache \
    --hicache-storage-backend juicefs \
    --hicache-storage-backend-config '{
        "mount_point": "/mnt/jfs",
        "use_direct_io": false
    }'
```

### Advanced Configuration

```bash
python -m sglang.launch_server \
    --model-path /data/Qwen/Qwen2.5-7B \
    --tp-size 2 \
    --pp-size 2 \
    --enable-hierarchical-cache \
    --hicache-ratio 2 \
    --hicache-mem-layout page_first \
    --hicache-write-policy write_through \
    --hicache-storage-backend juicefs \
    --hicache-storage-backend-config '{
        "mount_point": "/mnt/jfs",
        "use_direct_io": true,
        "enable_compression": false
    }' \
    --hicache-storage-prefetch-policy wait_complete
```

## Configuration Options

### Backend Configuration (`--hicache-storage-backend-config`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mount_point` | str | "/mnt/jfs" | JuiceFS mount point |
| `use_direct_io` | bool | false | Use direct I/O for better performance |
| `enable_compression` | bool | false | Enable data compression |

### HiCache Core Options

| Flag | Description |
|------|-------------|
| `--enable-hierarchical-cache` | Enable HiCache L1/L2/L3 caching |
| `--hicache-ratio` | Host memory ratio relative to GPU memory |
| `--hicache-mem-layout` | Memory layout (page_first/page_first_direct/layer_first) |
| `--hicache-write-policy` | Write policy (write_through/write_through_selective/write_back) |
| `--hicache-storage-prefetch-policy` | Prefetch strategy (best_effort/wait_complete/timeout) |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SGLang with HiCache                       │
├─────────────────────────────────────────────────────────────┤
│  L1: GPU KV Cache (Hot Data)                                │
│       ↓ (eviction)                                          │
│  L2: Host Memory (Warm Data)                                │
│       ↓ (write-back)                                        │
│  L3: JuiceFS Storage (Cold Data)                            │
│       │                                                     │
│       └──► /mnt/jfs (JuiceFS mount)                         │
│              ├─── Local Cache (/dev/vdc/jfs-cache)         │
│              └─── Backend Storage (S3/OSS/Local)           │
└─────────────────────────────────────────────────────────────┘
```

## Performance Tuning

### 1. JuiceFS Cache Configuration

```bash
# Mount with optimized cache settings
juicefs mount \
    --cache-dir /dev/vdc/jfs-cache \
    --cache-size 100000 \              # 100GB local cache
    --free-space-ratio 0.1 \           # Keep 10% free space
    --writeback \                      # Enable writeback mode
    --upload-delay 30 \                # Delay upload by 30s
    redis://localhost:6379/1 \
    /mnt/jfs
```

### 2. SGLang HiCache Configuration

```bash
# For high-throughput scenarios
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-mem-layout page_first_direct \
    --hicache-write-policy write_through \
    --hicache-storage-prefetch-policy best_effort \
    --hicache-storage-backend juicefs

# For high-cache-hit-rate scenarios  
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-mem-layout page_first_direct \
    --hicache-write-policy write_through_selective \
    --hicache-storage-prefetch-policy wait_complete \
    --hicache-storage-backend juicefs
```

## Monitoring

### Check JuiceFS Stats

```bash
# JuiceFS runtime stats
juicefs stats /mnt/jfs

# Check cache usage
du -sh /dev/vdc/jfs-cache
```

### Check HiCache Stats

Enable metrics in SGLang:
```bash
python -m sglang.launch_server \
    --enable-metrics \
    --enable-cache-report \
    ...
```

## Troubleshooting

### Issue: Backend not found

**Solution**: Ensure backend is registered:
```python
python register_juicefs_backend.py
```

### Issue: Permission denied on mount point

**Solution**: Check permissions:
```bash
sudo chown -R $(whoami):$(whoami) /mnt/jfs
```

### Issue: Poor performance

**Solutions**:
1. Enable local caching: `--cache-dir /dev/vdc/jfs-cache`
2. Use `page_first_direct` memory layout
3. Enable direct I/O: `"use_direct_io": true`
4. Tune `--cache-size` based on your workload

## API Reference

### HiCacheJuiceFS Class

```python
class HiCacheJuiceFS(HiCacheStorage):
    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        mount_point: str = "/mnt/jfs",
        use_direct_io: bool = False,
        enable_compression: bool = False,
        **kwargs
    )
    
    def get(self, key: str, target_location: torch.Tensor) -> torch.Tensor | None
    def set(self, key: str, target_location: torch.Tensor) -> bool
    def exists(self, key: str) -> bool
    def get_stats(self) -> dict
```

## License

Apache-2.0

## Contributing

Contributions are welcome! Please submit issues and pull requests.

## References

- [SGLang HiCache Documentation](https://docs.sglang.com.cn/advanced_features/hicache_design.html)
- [JuiceFS Documentation](https://juicefs.com/docs/community/introduction/)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
