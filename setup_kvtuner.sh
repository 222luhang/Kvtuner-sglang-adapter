#!/bin/bash
# KVTuner-SGLang Integration Setup Script
# This script sets up the KVTuner quantization adapter for SGLang

set -e

echo "=== KVTuner-SGLang Integration Setup ==="

# Check if we're in the right directory
if [ ! -f "python/sglang/srt/layers/quantization/kvtuner_quant.py" ]; then
    echo "Error: kvtuner_quant.py not found. Please run from sglang repo root."
    exit 1
fi

echo "1. Installing KVTuner flexible_quant package..."
cd /home/ubuntu/.openclaw/workspace/kvtuner/flexible_quant
pip install -e . -q

echo "2. Installing SGLang dependencies..."
cd /home/ubuntu/.openclaw/workspace/sglang
pip install -e "python[all]" -q 2>/dev/null || pip install -e python -q

echo "3. Applying KVTuner patches..."
python apply_kvtuner_patch.py

echo "4. Verifying installation..."
python -c "from sglang.srt.layers.quantization.kvtuner_quant import KVTunerQuantConfig; print('✓ KVTuner quantization module loaded successfully')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run SGLang with KVTuner quantization on 2 nodes (4 GPUs):"
echo ""
echo "Node 0 (10.60.252.76):"
echo 'python -m sglang.launch_server \\"
echo '  --model-path meta-llama/Llama-2-7b-hf \\"
echo '  --tp-size 4 \\"
echo '  --nnodes 2 \\"
echo '  --node-rank 0 \\"
echo '  --dist-init-addr "10.60.252.76:29500" \\"
echo '  --enable-kvtuner-quant \\"
echo '  --kvtuner-nbits-key 4 \\"
echo '  --kvtuner-nbits-value 4'
echo ""
echo "Node 1 (10.60.61.227):"
echo 'python -m sglang.launch_server \\"
echo '  --model-path meta-llama/Llama-2-7b-hf \\"
echo '  --tp-size 4 \\"
echo '  --nnodes 2 \\"
echo '  --node-rank 1 \\"
echo '  --dist-init-addr "10.60.252.76:29500" \\"
echo '  --enable-kvtuner-quant \\"
echo '  --kvtuner-nbits-key 4 \\"
echo '  --kvtuner-nbits-value 4'
