# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""
JuiceFS Storage Backend for SGLang HiCache

This module provides a JuiceFS-based storage backend for HiCache L3 layer.
JuiceFS is a high-performance distributed file system that can use object storage
as backend with local caching.

Usage:
    1. Mount JuiceFS to a local directory:
       juicefs mount <META-URL> /mnt/jfs --cache-dir /dev/vdc/jfs-cache
    
    2. Configure SGLang to use JuiceFS backend:
       --hicache-storage-backend juicefs
       --hicache-storage-backend-config '{"mount_point": "/mnt/jfs"}'
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


class HiCacheJuiceFS(HiCacheStorage):
    """
    JuiceFS-based storage backend for HiCache.
    
    Uses JuiceFS mounted directory as L3 storage for KV cache.
    Supports zero-copy operations when using Direct I/O.
    
    Args:
        storage_config: HiCacheStorageConfig with storage settings
        mount_point: JuiceFS mount point (default: /mnt/jfs)
        use_direct_io: Whether to use direct I/O for better performance
        enable_compression: Whether to enable data compression
    """

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        mount_point: str = "/mnt/jfs",
        use_direct_io: bool = False,
        enable_compression: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.storage_config = storage_config
        self.mount_point = Path(mount_point)
        self.use_direct_io = use_direct_io
        self.enable_compression = enable_compression
        
        # Build storage path with model and rank info
        model_name = storage_config.model_name or "default"
        model_name = model_name.replace("/", "-")
        
        if storage_config.is_mla_model:
            self.storage_suffix = f"{model_name}"
        else:
            self.storage_suffix = f"{model_name}_tp{storage_config.tp_rank}_{storage_config.tp_size}"
        
        self.storage_dir = self.mount_point / "hicache" / self.storage_suffix
        
        # Create directories
        self._ensure_directories()
        
        # Statistics
        self.stats = {
            "get_count": 0,
            "set_count": 0,
            "exists_count": 0,
            "hit_count": 0,
            "miss_count": 0,
            "bytes_read": 0,
            "bytes_written": 0,
        }
        
        logger.info(
            f"HiCacheJuiceFS initialized: mount_point={mount_point}, "
            f"storage_dir={self.storage_dir}, use_direct_io={use_direct_io}"
        )

    def _ensure_directories(self) -> None:
        """Create necessary directories."""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for sharding (256 shards based on key prefix)
            for i in range(256):
                (self.storage_dir / f"shard_{i:02x}").mkdir(exist_ok=True)
                
            logger.info(f"Created JuiceFS storage directory: {self.storage_dir}")
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            raise

    def _get_shard_path(self, key: str) -> Path:
        """Get shard directory for a key."""
        # Use first 2 chars of key hash for sharding
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        shard_id = key_hash[:2]
        return self.storage_dir / f"shard_{shard_id}"

    def _get_file_path(self, key: str) -> Path:
        """Get file path for a key."""
        return self._get_shard_path(key) / f"{key}.bin"

    def _get_metadata_path(self, key: str) -> Path:
        """Get metadata file path for a key."""
        return self._get_shard_path(key) / f"{key}.meta"

    def get(
        self,
        key: str,
        target_location: Optional[torch.Tensor] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        Retrieve KV cache data from JuiceFS.
        
        Args:
            key: Unique identifier for the KV cache
            target_location: Pre-allocated tensor to store the data
            target_sizes: Optional size information
            
        Returns:
            The tensor with loaded data, or None if key not found
        """
        self.stats["get_count"] += 1
        
        file_path = self._get_file_path(key)
        meta_path = self._get_metadata_path(key)
        
        try:
            # Check if file exists
            if not file_path.exists():
                self.stats["miss_count"] += 1
                logger.debug(f"Key {key} not found in JuiceFS storage")
                return None
            
            # Read metadata
            metadata = {}
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
            
            # Read data
            if target_location is not None:
                # Read directly into pre-allocated tensor (zero-copy)
                expected_bytes = target_location.numel() * target_location.element_size()
                
                with open(file_path, 'rb') as f:
                    buf = target_location.view(torch.uint8).contiguous().numpy()
                    bytes_read = f.readinto(buf)
                    
                    if bytes_read != expected_bytes:
                        raise IOError(f"Short read: expected {expected_bytes}, got {bytes_read}")
                
                self.stats["bytes_read"] += bytes_read
                self.stats["hit_count"] += 1
                
                logger.debug(f"Loaded {key} from JuiceFS: {bytes_read} bytes")
                return target_location
            else:
                # Load into new tensor
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                # Deserialize based on metadata
                dtype = getattr(torch, metadata.get('dtype', 'float16'))
                shape = metadata.get('shape')
                
                tensor = torch.frombuffer(data, dtype=dtype).reshape(shape)
                self.stats["bytes_read"] += len(data)
                self.stats["hit_count"] += 1
                
                return tensor
                
        except FileNotFoundError:
            self.stats["miss_count"] += 1
            logger.debug(f"Key {key} not found")
            return None
        except Exception as e:
            logger.error(f"Error reading {key}: {e}")
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor]] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        """
        Batch retrieve multiple KV cache entries.
        
        Args:
            keys: List of keys to retrieve
            target_locations: List of pre-allocated tensors
            target_sizes: Optional size information
            
        Returns:
            List of tensors or None for each key
        """
        if target_locations is None:
            target_locations = [None] * len(keys)
        
        results = []
        for key, target in zip(keys, target_locations):
            result = self.get(key, target)
            results.append(result)
        
        return results

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[torch.Tensor] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store KV cache data to JuiceFS.
        
        Args:
            key: Unique identifier for the KV cache
            value: Data to store (optional if target_location provided)
            target_location: Tensor containing data to store
            target_sizes: Optional size information
            
        Returns:
            True if successful, False otherwise
        """
        self.stats["set_count"] += 1
        
        # Skip if already exists (HiCache handles deduplication)
        if self.exists(key):
            logger.debug(f"Key {key} already exists, skipping")
            return True
        
        try:
            # Determine data source
            if target_location is not None:
                data_tensor = target_location
            elif value is not None:
                data_tensor = value
            else:
                logger.error("No data provided for set operation")
                return False
            
            # Ensure tensor is contiguous and on CPU
            if isinstance(data_tensor, torch.Tensor):
                data_tensor = data_tensor.contiguous().cpu()
                data_bytes = data_tensor.view(torch.uint8).numpy().tobytes()
                
                # Store metadata
                metadata = {
                    "dtype": str(data_tensor.dtype).replace("torch.", ""),
                    "shape": list(data_tensor.shape),
                    "size_bytes": len(data_bytes),
                    "timestamp": time.time(),
                }
            else:
                data_bytes = data_tensor
                metadata = {
                    "size_bytes": len(data_bytes),
                    "timestamp": time.time(),
                }
            
            file_path = self._get_file_path(key)
            meta_path = self._get_metadata_path(key)
            
            # Write data
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if self.use_direct_io:
                flags |= os.O_DIRECT
            
            with open(file_path, 'wb') as f:
                f.write(data_bytes)
            
            # Write metadata
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
            
            self.stats["bytes_written"] += len(data_bytes)
            
            logger.debug(f"Stored {key} to JuiceFS: {len(data_bytes)} bytes")
            return True
            
        except Exception as e:
            logger.error(f"Error writing {key}: {e}")
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Batch store multiple KV cache entries.
        
        Args:
            keys: List of keys to store
            values: Data to store
            target_locations: Source tensor locations
            target_sizes: Optional size information
            
        Returns:
            True if all operations successful
        """
        success = True
        
        if target_locations is not None:
            for key, target in zip(keys, target_locations):
                if not self.set(key, target_location=target):
                    success = False
        elif values is not None:
            for key, value in zip(keys, values):
                if not self.set(key, value=value):
                    success = False
        
        return success

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in storage.
        
        Args:
            key: Key to check
            
        Returns:
            True if exists, False otherwise
        """
        self.stats["exists_count"] += 1
        return self._get_file_path(key).exists()

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """
        Check if multiple keys exist.
        Returns the number of consecutive existing keys from start.
        """
        for i, key in enumerate(keys):
            if not self.exists(key):
                return i
        return len(keys)

    def clear(self) -> None:
        """Clear all stored data."""
        try:
            import shutil
            if self.storage_dir.exists():
                shutil.rmtree(self.storage_dir)
                self._ensure_directories()
            
            # Reset stats
            for key in self.stats:
                self.stats[key] = 0
                
            logger.info("JuiceFS storage cleared")
        except Exception as e:
            logger.error(f"Error clearing storage: {e}")

    def get_stats(self) -> dict:
        """Get storage statistics."""
        hit_rate = 0
        if self.stats["get_count"] > 0:
            hit_rate = self.stats["hit_count"] / self.stats["get_count"]
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "storage_dir": str(self.storage_dir),
            "mount_point": str(self.mount_point),
        }

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        """Register host memory pool for optimized operations."""
        super().register_mem_pool_host(mem_pool_host)
        logger.info(f"Registered memory pool host with layout: {mem_pool_host.layout}")


# Factory function for dynamic loading
def create_juicefs_backend(
    storage_config: HiCacheStorageConfig,
    **kwargs
) -> HiCacheJuiceFS:
    """
    Factory function for creating JuiceFS backend.
    Supports dynamic loading from SGLang.
    """
    # Parse extra config if provided
    extra = storage_config.extra_config or {}
    
    # Merge configs (kwargs take precedence)
    config = {**extra, **kwargs}
    
    return HiCacheJuiceFS(storage_config, **config)
