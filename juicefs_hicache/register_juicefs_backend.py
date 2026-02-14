#!/usr/bin/env python3
"""
Register JuiceFS backend with SGLang HiCache

This script registers the JuiceFS storage backend with SGLang's 
storage backend factory, enabling dynamic loading.
"""

import sys
import os

# Add the directory containing juicefs_hicache_storage.py to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

def register_juicefs_backend():
    """Register JuiceFS backend with SGLang."""
    try:
        from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
        
        # Register the JuiceFS backend
        StorageBackendFactory.register_backend(
            name="juicefs",
            module_path="juicefs_hicache_storage",
            class_name="HiCacheJuiceFS"
        )
        
        print("✅ JuiceFS backend registered successfully!")
        print("   Backend name: juicefs")
        print("   Module: juicefs_hicache_storage")
        print("   Class: HiCacheJuiceFS")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import SGLang: {e}")
        print("   Make sure SGLang is installed in your environment.")
        return False
    except Exception as e:
        print(f"❌ Error registering backend: {e}")
        return False


if __name__ == "__main__":
    success = register_juicefs_backend()
    sys.exit(0 if success else 1)
