#!/bin/bash
# Example: Launch SGLang with JuiceFS HiCache Backend

set -e

echo "=============================================="
echo "SGLang + JuiceFS HiCache Backend Example"
echo "=============================================="

# Configuration
MODEL_PATH="/data/Qwen/Qwen2.5-7B"
JUICE_MOUNT="/mnt/jfs"
CACHE_DIR="/dev/vdc/jfs-cache"

echo ""
echo "Step 1: Check JuiceFS Mount"
echo "------------------------------"
if mount | grep -q "$JUICE_MOUNT"; then
    echo "✅ JuiceFS is mounted at $JUICE_MOUNT"
    df -h "$JUICE_MOUNT"
else
    echo "⚠️  JuiceFS is not mounted at $JUICE_MOUNT"
    echo "   Please mount JuiceFS first:"
    echo "   juicefs mount redis://localhost:6379/1 $JUICE_MOUNT --cache-dir $CACHE_DIR"
    exit 1
fi

echo ""
echo "Step 2: Register JuiceFS Backend"
echo "---------------------------------"
python3 register_juicefs_backend.py || {
    echo "❌ Failed to register backend"
    exit 1
}

echo ""
echo "Step 3: Launch SGLang with JuiceFS Backend"
echo "--------------------------------------------"

# Option 1: Using registered backend
python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tp-size 2 \
    --pp-size 1 \
    --host 0.0.0.0 \
    --port 30000 \
    --enable-hierarchical-cache \
    --hicache-ratio 2 \
    --hicache-mem-layout page_first_direct \
    --hicache-write-policy write_through \
    --hicache-storage-backend juicefs \
    --hicache-storage-backend-config "{
        \"mount_point\": \"$JUICE_MOUNT\",
        \"use_direct_io\": false,
        \"enable_compression\": false
    }" \
    --hicache-storage-prefetch-policy wait_complete \
    --enable-metrics \
    --enable-cache-report \
    --log-level info

# Option 2: Using dynamic backend loading (alternative)
# python3 -m sglang.launch_server \
#     --model-path "$MODEL_PATH" \
#     --tp-size 2 \
#     --enable-hierarchical-cache \
#     --hicache-storage-backend dynamic \
#     --hicache-storage-backend-config '{
#         "backend_name": "juicefs",
#         "module_path": "juicefs_hicache_storage",
#         "class_name": "HiCacheJuiceFS",
#         "mount_point": "/mnt/jfs",
#         "use_direct_io": false
#     }'
