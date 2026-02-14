#!/usr/bin/env python3
"""
Test script for JuiceFS HiCache Storage Backend
"""

import sys
import os
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    from juicefs_hicache_storage import HiCacheJuiceFS, create_juicefs_backend
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Please ensure SGLang is installed.")
    sys.exit(1)


def test_basic_operations():
    """Test basic get/set/exists operations."""
    print("\n" + "="*60)
    print("Testing JuiceFS HiCache Backend - Basic Operations")
    print("="*60)
    
    # Create test directory
    test_dir = tempfile.mkdtemp(prefix="juicefs_test_")
    print(f"\nTest directory: {test_dir}")
    
    # Create storage config
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        is_mla_model=False,
        is_page_first_layout=True,
        model_name="test-model",
        extra_config={}
    )
    
    # Create backend
    try:
        backend = HiCacheJuiceFS(
            storage_config=config,
            mount_point=test_dir,
            use_direct_io=False
        )
        print("✅ Backend created successfully")
    except Exception as e:
        print(f"❌ Failed to create backend: {e}")
        return False
    
    # Test set
    print("\n--- Testing SET operation ---")
    key = "test_key_001"
    test_data = torch.randn(10, 20, dtype=torch.float16)
    
    try:
        success = backend.set(key, target_location=test_data)
        if success:
            print(f"✅ Set operation successful: {key}")
        else:
            print(f"❌ Set operation failed")
            return False
    except Exception as e:
        print(f"❌ Set operation error: {e}")
        return False
    
    # Test exists
    print("\n--- Testing EXISTS operation ---")
    try:
        exists = backend.exists(key)
        if exists:
            print(f"✅ Exists check passed: {key} exists")
        else:
            print(f"❌ Exists check failed: {key} not found")
            return False
    except Exception as e:
        print(f"❌ Exists operation error: {e}")
        return False
    
    # Test get
    print("\n--- Testing GET operation ---")
    try:
        target = torch.empty(10, 20, dtype=torch.float16)
        result = backend.get(key, target_location=target)
        
        if result is not None:
            # Verify data
            if torch.allclose(test_data, result):
                print(f"✅ Get operation successful: data verified")
            else:
                print(f"❌ Data mismatch!")
                return False
        else:
            print(f"❌ Get operation returned None")
            return False
    except Exception as e:
        print(f"❌ Get operation error: {e}")
        return False
    
    # Test stats
    print("\n--- Testing STATS ---")
    try:
        stats = backend.get_stats()
        print(f"Backend stats:")
        print(f"  - Get count: {stats.get('get_count', 0)}")
        print(f"  - Set count: {stats.get('set_count', 0)}")
        print(f"  - Hit count: {stats.get('hit_count', 0)}")
        print(f"  - Hit rate: {stats.get('hit_rate', 0):.2%}")
        print(f"  - Bytes read: {stats.get('bytes_read', 0)}")
        print(f"  - Bytes written: {stats.get('bytes_written', 0)}")
    except Exception as e:
        print(f"⚠️  Stats error: {e}")
    
    # Cleanup
    print("\n--- Cleanup ---")
    try:
        backend.clear()
        print("✅ Cleanup completed")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")
    
    # Remove test directory
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    
    print("\n" + "="*60)
    print("✅ All basic tests passed!")
    print("="*60)
    return True


def test_batch_operations():
    """Test batch operations."""
    print("\n" + "="*60)
    print("Testing JuiceFS HiCache Backend - Batch Operations")
    print("="*60)
    
    test_dir = tempfile.mkdtemp(prefix="juicefs_batch_test_")
    
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        is_mla_model=False,
        is_page_first_layout=True,
        model_name="test-model",
        extra_config={}
    )
    
    backend = HiCacheJuiceFS(
        storage_config=config,
        mount_point=test_dir,
        use_direct_io=False
    )
    
    # Test batch set
    print("\n--- Testing BATCH SET ---")
    keys = [f"batch_key_{i:03d}" for i in range(10)]
    values = [torch.randn(10, 20, dtype=torch.float16) for _ in range(10)]
    
    try:
        success = backend.batch_set(keys, values=values)
        if success:
            print(f"✅ Batch set successful: {len(keys)} items")
        else:
            print(f"❌ Batch set failed")
            return False
    except Exception as e:
        print(f"❌ Batch set error: {e}")
        return False
    
    # Test batch exists
    print("\n--- Testing BATCH EXISTS ---")
    try:
        count = backend.batch_exists(keys)
        print(f"✅ Batch exists: {count}/{len(keys)} keys found")
    except Exception as e:
        print(f"❌ Batch exists error: {e}")
        return False
    
    # Cleanup
    backend.clear()
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    
    print("\n" + "="*60)
    print("✅ All batch tests passed!")
    print("="*60)
    return True


def test_factory_function():
    """Test factory function."""
    print("\n" + "="*60)
    print("Testing Factory Function")
    print("="*60)
    
    test_dir = tempfile.mkdtemp(prefix="juicefs_factory_test_")
    
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
        is_mla_model=False,
        is_page_first_layout=True,
        model_name="test-model",
        extra_config={
            "mount_point": test_dir,
            "use_direct_io": False
        }
    )
    
    try:
        backend = create_juicefs_backend(config)
        print("✅ Factory function created backend successfully")
        print(f"   Mount point: {backend.mount_point}")
        print(f"   Storage dir: {backend.storage_dir}")
    except Exception as e:
        print(f"❌ Factory function error: {e}")
        return False
    finally:
        backend.clear()
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
    
    print("\n" + "="*60)
    print("✅ Factory function test passed!")
    print("="*60)
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("JuiceFS HiCache Backend Test Suite")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Basic Operations", test_basic_operations()))
    results.append(("Batch Operations", test_batch_operations()))
    results.append(("Factory Function", test_factory_function()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        sys.exit(1)
