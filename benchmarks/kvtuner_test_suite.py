#!/usr/bin/env python3
"""
KVTuner-SGLang Comprehensive Test Suite
Tests for thesis: Ablation studies, TTFT, Throughput, Memory usage
"""

import requests
import time
import json
import statistics
from datetime import datetime
import sys

# Configuration
BASE_URL = "http://localhost:30001"
MODEL_PATH = "/data/Qwen/Qwen2.5-7B"

# Test prompts of varying lengths
TEST_PROMPTS = {
    "short": "你好，请介绍一下自己",
    "medium": "人工智能技术在自然语言处理领域取得了重大进展。请详细说明Transformer架构的工作原理及其在机器翻译中的应用。",
    "long": "深度学习是机器学习的一个分支，它基于人工神经网络。深度学习模型通过多层非线性变换来学习数据的层次化表示。" * 10,
}

# Quantization configs to test
QUANT_CONFIGS = [
    {"name": "FP16_baseline", "enable_kvtuner_quant": False},
    {"name": "KV8", "enable_kvtuner_quant": True, "kvtuner_nbits_key": 8, "kvtuner_nbits_value": 8},
    {"name": "KV4", "enable_kvtuner_quant": True, "kvtuner_nbits_key": 4, "kvtuner_nbits_value": 4},
    {"name": "KV2", "enable_kvtuner_quant": True, "kvtuner_nbits_key": 2, "kvtuner_nbits_value": 2},
]

class KVTunerTester:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
        self.results = []
        
    def check_health(self):
        """Check if service is ready"""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except:
            return False
    
    def get_model_info(self):
        """Get model and quantization info"""
        try:
            r = requests.get(f"{self.base_url}/model_info", timeout=10)
            return r.json()
        except Exception as e:
            return {"error": str(e)}
    
    def generate(self, prompt, max_tokens=50, temperature=0.7):
        """Generate text and measure metrics"""
        start_time = time.time()
        
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_tokens
            }
        }
        
        try:
            r = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=120
            )
            result = r.json()
            
            end_time = time.time()
            latency = end_time - start_time
            
            return {
                "success": True,
                "latency": latency,
                "prompt_tokens": len(prompt) // 4,  # Rough estimate
                "completion_tokens": max_tokens,
                "output": result.get("text", ""),
                "meta_info": result.get("meta_info", {})
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "latency": time.time() - start_time
            }
    
    def test_ttft(self, prompt_lengths=[10, 50, 100, 500, 1000]):
        """Test Time To First Token with varying prompt lengths"""
        print(f"\n{'='*60}")
        print("TEST: Time To First Token (TTFT)")
        print(f"{'='*60}")
        
        results = []
        for length in prompt_lengths:
            prompt = "测" * length
            
            # Warmup
            self.generate(prompt[:10], max_tokens=1)
            
            # Measure TTFT
            latencies = []
            for _ in range(3):  # 3 runs for average
                result = self.generate(prompt, max_tokens=1)
                if result["success"]:
                    latencies.append(result["latency"])
                time.sleep(0.5)
            
            if latencies:
                avg_latency = statistics.mean(latencies)
                results.append({
                    "prompt_length": length,
                    "avg_ttft_ms": avg_latency * 1000,
                    "min_ms": min(latencies) * 1000,
                    "max_ms": max(latencies) * 1000
                })
                print(f"  Prompt length {length:4d}: TTFT = {avg_latency*1000:.2f} ms")
        
        return results
    
    def test_throughput(self, concurrent_requests=[1, 2, 4, 8]):
        """Test throughput with concurrent requests"""
        print(f"\n{'='*60}")
        print("TEST: Throughput")
        print(f"{'='*60}")
        
        import threading
        
        results = []
        prompt = "人工智能的发展对未来社会有什么影响？"
        
        for concurrency in concurrent_requests:
            latencies = []
            errors = 0
            
            def worker():
                try:
                    result = self.generate(prompt, max_tokens=50)
                    if result["success"]:
                        latencies.append(result["latency"])
                except:
                    nonlocal errors
                    errors += 1
            
            # Launch concurrent requests
            threads = []
            start_time = time.time()
            
            for _ in range(concurrency):
                t = threading.Thread(target=worker)
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            total_time = time.time() - start_time
            
            if latencies:
                throughput = len(latencies) / total_time
                avg_latency = statistics.mean(latencies)
                results.append({
                    "concurrency": concurrency,
                    "throughput_req_per_sec": throughput,
                    "avg_latency_ms": avg_latency * 1000,
                    "success_rate": len(latencies) / concurrency
                })
                print(f"  Concurrency {concurrency}: {throughput:.2f} req/s, "
                      f"avg latency = {avg_latency*1000:.2f} ms")
        
        return results
    
    def test_memory_usage(self):
        """Get memory usage statistics"""
        print(f"\n{'='*60}")
        print("TEST: Memory Usage")
        print(f"{'='*60}")
        
        # Generate some load to fill KV cache
        print("  Generating load to fill KV cache...")
        for _ in range(5):
            self.generate("测试内存使用情况的提示词。" * 20, max_tokens=100)
            time.sleep(0.5)
        
        # Get memory info from nvidia-smi
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            memory_info = result.stdout.strip().split('\n')
            print(f"  GPU Memory Usage:")
            for i, info in enumerate(memory_info):
                used, total = info.split(', ')
                print(f"    GPU {i}: {used} MiB / {total} MiB ({float(used)/float(total)*100:.1f}%)")
        except Exception as e:
            print(f"  Error getting memory info: {e}")
        
        return {}
    
    def run_all_tests(self):
        """Run complete test suite"""
        print(f"\n{'='*60}")
        print("KVTuner-SGLang Comprehensive Test Suite")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {MODEL_PATH}")
        print(f"{'='*60}")
        
        # Check service health
        if not self.check_health():
            print("❌ Service not ready!")
            return None
        
        print("✅ Service is ready")
        
        # Get model info
        model_info = self.get_model_info()
        print(f"\nModel Info: {json.dumps(model_info, indent=2, ensure_ascii=False)}")
        
        # Run tests
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_PATH,
            "ttft": self.test_ttft(),
            "throughput": self.test_throughput(),
            "memory": self.test_memory_usage(),
        }
        
        # Save results
        output_file = f"/tmp/kvtuner_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"✅ Tests completed!")
        print(f"Results saved to: {output_file}")
        print(f"{'='*60}")
        
        return results

if __name__ == "__main__":
    tester = KVTunerTester()
    tester.run_all_tests()
