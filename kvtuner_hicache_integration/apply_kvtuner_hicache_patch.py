#!/usr/bin/env python3
"""
Apply KVTuner-HiCache Integration Patch to SGLang

This script patches SGLang to enable KVTuner quantization for HiCache.
"""

import os
import sys


def patch_hiradix_cache():
    """Patch HiRadixCache to use KVTuner-enabled host pools."""
    
    hiradix_path = "python/sglang/srt/mem_cache/hiradix_cache.py"
    
    with open(hiradix_path, 'r') as f:
        content = f.read()
    
    # Add import for KVTuner host pools
    import_addition = '''from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig

# KVTuner HiCache Integration
try:
    from sglang.srt.kvtuner_hicache_integration import (
        KVTunerMHATokenToKVPoolHost,
        KVTunerMLATokenToKVPoolHost,
    )
    KVTUNER_HICACHE_AVAILABLE = True
except ImportError:
    KVTUNER_HICACHE_AVAILABLE = False'''
    
    content = content.replace(
        'from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig',
        import_addition
    )
    
    # Replace MHATokenToKVPoolHost initialization
    old_mha_init = '''if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )'''
    
    new_mha_init = '''if isinstance(self.kv_cache, MHATokenToKVPool):
            if KVTUNER_HICACHE_AVAILABLE and getattr(server_args, 'enable_kvtuner_hicache', False):
                self.token_to_kv_pool_host = KVTunerMHATokenToKVPoolHost(
                    self.kv_cache,
                    server_args.hicache_ratio,
                    server_args.hicache_size,
                    self.page_size,
                    server_args.hicache_mem_layout,
                    allocator_type=server_args.hicache_storage_backend,
                    enable_kvtuner=True,
                    kvtuner_nbits_key=getattr(server_args, 'kvtuner_hicache_nbits_key', 4),
                    kvtuner_nbits_value=getattr(server_args, 'kvtuner_hicache_nbits_value', 4),
                    kvtuner_asym=getattr(server_args, 'kvtuner_hicache_asym', False),
                    kvtuner_axis_key=getattr(server_args, 'kvtuner_hicache_axis_key', 0),
                    kvtuner_axis_value=getattr(server_args, 'kvtuner_hicache_axis_value', 0),
                    kvtuner_q_group_size=getattr(server_args, 'kvtuner_hicache_q_group_size', 64),
                    kvtuner_residual_length=getattr(server_args, 'kvtuner_hicache_residual_length', 128),
                )
            else:
                self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                    self.kv_cache,
                    server_args.hicache_ratio,
                    server_args.hicache_size,
                    self.page_size,
                    server_args.hicache_mem_layout,
                    allocator_type=server_args.hicache_storage_backend,
                )'''
    
    content = content.replace(old_mha_init, new_mha_init)
    
    # Replace MLATokenToKVPoolHost initialization
    old_mla_init = '''elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )'''
    
    new_mla_init = '''elif isinstance(self.kv_cache, MLATokenToKVPool):
            if KVTUNER_HICACHE_AVAILABLE and getattr(server_args, 'enable_kvtuner_hicache', False):
                self.token_to_kv_pool_host = KVTunerMLATokenToKVPoolHost(
                    self.kv_cache,
                    server_args.hicache_ratio,
                    server_args.hicache_size,
                    self.page_size,
                    server_args.hicache_mem_layout,
                    allocator_type=server_args.hicache_storage_backend,
                    enable_kvtuner=True,
                    kvtuner_nbits_key=getattr(server_args, 'kvtuner_hicache_nbits_key', 4),
                    kvtuner_nbits_value=getattr(server_args, 'kvtuner_hicache_nbits_value', 4),
                    kvtuner_asym=getattr(server_args, 'kvtuner_hicache_asym', False),
                    kvtuner_axis_key=getattr(server_args, 'kvtuner_hicache_axis_key', 0),
                    kvtuner_axis_value=getattr(server_args, 'kvtuner_hicache_axis_value', 0),
                    kvtuner_q_group_size=getattr(server_args, 'kvtuner_hicache_q_group_size', 64),
                    kvtuner_residual_length=getattr(server_args, 'kvtuner_hicache_residual_length', 128),
                )
            else:
                self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                    self.kv_cache,
                    server_args.hicache_ratio,
                    server_args.hicache_size,
                    self.page_size,
                    server_args.hicache_mem_layout,
                    allocator_type=server_args.hicache_storage_backend,
                )'''
    
    content = content.replace(old_mla_init, new_mla_init)
    
    with open(hiradix_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Patched {hiradix_path}")


def patch_server_args():
    """Add KVTuner HiCache arguments to server_args.py."""
    
    server_args_path = "python/sglang/srt/server_args.py"
    
    with open(server_args_path, 'r') as f:
        content = f.read()
    
    # Add KVTuner HiCache config fields after existing KVTuner fields
    kvtuner_hicache_fields = '''
    # KVTuner HiCache Integration
    enable_kvtuner_hicache: bool = False
    kvtuner_hicache_nbits_key: int = 4
    kvtuner_hicache_nbits_value: int = 4
    kvtuner_hicache_asym: bool = False
    kvtuner_hicache_axis_key: int = 0
    kvtuner_hicache_axis_value: int = 0
    kvtuner_hicache_q_group_size: int = 64
    kvtuner_hicache_residual_length: int = 128
'''
    
    # Find the existing KVTuner fields and add after them
    marker = 'kvtuner_residual_length: int = 128'
    if marker in content:
        content = content.replace(
            marker,
            marker + kvtuner_hicache_fields
        )
    
    # Add CLI arguments
    kvtuner_hicache_cli = '''
        # KVTuner HiCache Integration
        parser.add_argument(
            "--enable-kvtuner-hicache",
            action="store_true",
            help="Enable KVTuner quantization for HiCache L2/L3 storage.",
        )
        parser.add_argument(
            "--kvtuner-hicache-nbits-key",
            type=int,
            default=4,
            choices=[2, 4, 8],
            help="Number of bits for key quantization in HiCache.",
        )
        parser.add_argument(
            "--kvtuner-hicache-nbits-value",
            type=int,
            default=4,
            choices=[2, 4, 8],
            help="Number of bits for value quantization in HiCache.",
        )
        parser.add_argument(
            "--kvtuner-hicache-asym",
            action="store_true",
            help="Use asymmetric quantization for HiCache.",
        )
        parser.add_argument(
            "--kvtuner-hicache-axis-key",
            type=int,
            default=0,
            choices=[0, 1],
            help="Quantization axis for keys in HiCache (0=token, 1=channel).",
        )
        parser.add_argument(
            "--kvtuner-hicache-axis-value",
            type=int,
            default=0,
            choices=[0, 1],
            help="Quantization axis for values in HiCache (0=token, 1=channel).",
        )
        parser.add_argument(
            "--kvtuner-hicache-q-group-size",
            type=int,
            default=64,
            help="Group size for quantization in HiCache.",
        )
        parser.add_argument(
            "--kvtuner-hicache-residual-length",
            type=int,
            default=128,
            help="Number of recent tokens to keep in full precision in HiCache.",
        )
'''
    
    # Find a good place to insert CLI args (after existing KVTuner args)
    marker = '--kvtuner-residual-length'
    if f'"{marker}"' in content:
        # Find the end of kvtuner-residual-length argument
        lines = content.split('\n')
        insert_idx = None
        for i, line in enumerate(lines):
            if marker in line and 'help=' in lines[i+2]:
                # Find the closing parenthesis of this argument
                for j in range(i, min(i+10, len(lines))):
                    if lines[j].strip() == ')':
                        insert_idx = j + 1
                        break
                break
        
        if insert_idx:
            lines.insert(insert_idx, kvtuner_hicache_cli)
            content = '\n'.join(lines)
    
    with open(server_args_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Patched {server_args_path}")


if __name__ == "__main__":
    sglang_root = os.environ.get('SGLANG_ROOT', '/home/ubuntu/sglang-repo')
    os.chdir(sglang_root)
    
    print("Applying KVTuner-HiCache Integration Patch...")
    print()
    
    # Backup files
    os.system("cp python/sglang/srt/mem_cache/hiradix_cache.py python/sglang/srt/mem_cache/hiradix_cache.py.bak")
    os.system("cp python/sglang/srt/server_args.py python/sglang/srt/server_args.py.bak")
    
    # Apply patches
    patch_hiradix_cache()
    patch_server_args()
    
    print()
    print("✅ KVTuner-HiCache Integration patch applied successfully!")
    print()
    print("New CLI arguments added:")
    print("  --enable-kvtuner-hicache")
    print("  --kvtuner-hicache-nbits-key")
    print("  --kvtuner-hicache-nbits-value")
    print("  --kvtuner-hicache-asym")
    print("  --kvtuner-hicache-axis-key")
    print("  --kvtuner-hicache-axis-value")
    print("  --kvtuner-hicache-q-group-size")
    print("  --kvtuner-hicache-residual-length")
