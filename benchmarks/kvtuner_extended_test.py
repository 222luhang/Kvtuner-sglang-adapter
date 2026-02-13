#!/usr/bin/env python3
"""
Extended KVTuner-SGLang Test Suite
Comprehensive tests for thesis: Long sequences, batch tests, stability tests
"""

import requests
import time
import json
import statistics
import threading
from datetime import datetime
import sys

BASE_URL = "http://localhost:30001"
MODEL_PATH = "/data/Qwen/Qwen2.5-7B"

class ExtendedTester:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
        self.results = {}
        
    def check_health(self):
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except:
            return False
    
    def generate(self, prompt, max_tokens=50, temperature=0.7):
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
            
            return {
                "success": True,
                "latency": end_time - start_time,
                "output": result.get("text", ""),
                "meta_info": result.get("meta_info", {})
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "latency": time.time() - start_time
            }
    
    def test_long_sequences(self, seq_lengths=[100, 500, 1000, 2000, 4000]):
        """Test with varying sequence lengths"""
        print(f"\n{'='*60}")
        print("TEST: Long Sequence Handling")
        print(f"{'='*60}")
        
        results = []
        for length in seq_lengths:
            prompt = "测" * length
            
            # Warmup
            self.generate(prompt[:10], max_tokens=10)
            
            # Test generation with varying output lengths
            test_cases = [
                ("short_output", 50),
                ("medium_output", 200),
                ("long_output", 500)
            ]
            
            for case_name, max_tokens in test_cases:
                latencies = []
                for _ in range(3):
                    result = self.generate(prompt, max_tokens=max_tokens)
                    if result["success"]:
                        latencies.append(result["latency"])
                    time.sleep(0.5)
                
                if latencies:
                    results.append({
                        "seq_length": length,
                        "output_type": case_name,
                        "max_tokens": max_tokens,
                        "avg_latency_ms": statistics.mean(latencies) * 1000,
                        "throughput_tokens_per_sec": max_tokens / statistics.mean(latencies)
                    })
                    print(f"  Seq={length}, Output={max_tokens}: {statistics.mean(latencies)*1000:.2f}ms, "
                          f"{max_tokens/statistics.mean(latencies):.2f} tok/s")
        
        return results
    
    def test_batch_processing(self, batch_sizes=[1, 2, 4, 8, 16]):
        """Test batch processing efficiency"""
        print(f"\n{'='*60}")
        print("TEST: Batch Processing")
        print(f"{'='*60}")
        
        results = []
        prompt = "请详细解释机器学习中的注意力机制原理及其在Transformer中的应用。"
        
        for batch_size in batch_sizes:
            latencies = []
            threads = []
            
            def worker():
                result = self.generate(prompt, max_tokens=100)
                if result["success"]:
                    latencies.append(result["latency"])
            
            start_time = time.time()
            
            # Launch batch_size concurrent requests
            for _ in range(batch_size):
                t = threading.Thread(target=worker)
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            total_time = time.time() - start_time
            
            if latencies:
                results.append({
                    "batch_size": batch_size,
                    "total_time_ms": total_time * 1000,
                    "avg_latency_ms": statistics.mean(latencies) * 1000,
                    "throughput_req_per_sec": batch_size / total_time,
                    "success_count": len(latencies)
                })
                print(f"  Batch={batch_size}: {total_time*1000:.2f}ms total, "
                      f"{batch_size/total_time:.2f} req/s")
        
        return results
    
    def test_stability(self, duration_seconds=300, interval_seconds=10):
        """Long-running stability test"""
        print(f"\n{'='*60}")
        print(f"TEST: Stability Test ({duration_seconds}s)")
        print(f"{'='*60}")
        
        results = []
        prompt = "这是一个稳定性测试提示词。"
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < duration_seconds:
            iteration += 1
            result = self.generate(prompt, max_tokens=50)
            
            results.append({
                "iteration": iteration,
                "timestamp": time.time() - start_time,
                "success": result["success"],
                "latency_ms": result["latency"] * 1000 if result["success"] else None,
                "error": result.get("error") if not result["success"] else None
            })
            
            if iteration % 10 == 0:
                success_rate = sum(1 for r in results if r["success"]) / len(results)
                avg_latency = statistics.mean([r["latency_ms"] for r in results if r["latency_ms"]])
                print(f"  Iter {iteration}: Success={success_rate*100:.1f}%, Avg Latency={avg_latency:.2f}ms")
            
            time.sleep(interval_seconds)
        
        return results
    
    def test_memory_stress(self, iterations=50):
        """Memory stress test with continuous generation"""
        print(f"\n{'='*60}")
        print("TEST: Memory Stress Test")
        print(f"{'='*60}")
        
        import subprocess
        
        results = []
        prompt = "这是一个内存压力测试。" * 50  # Long prompt
        
        for i in range(iterations):
            result = self.generate(prompt, max_tokens=200)
            
            # Get memory usage
            try:
                mem_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                mem_usage = [int(x.strip()) for x in mem_result.stdout.strip().split('\n')]
            except:
                mem_usage = []
            
            results.append({
                "iteration": i,
                "success": result["success"],
                "latency_ms": result["latency"] * 1000 if result["success"] else None,
                "memory_mib": mem_usage
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i+1}/{iterations}: Memory={mem_usage}")
        
        return results
    
    def run_all_extended_tests(self):
        """Run all extended tests"""
        print(f"\n{'='*60}")
        print("KVTuner-SGLang Extended Test Suite")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {MODEL_PATH}")
        print(f"{'='*60}")
        
        if not self.check_health():
            print("❌ Service not ready!")
            return None
        
        print("✅ Service is ready")
        
        # Run tests
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_PATH,
            "long_sequences": self.test_long_sequences(),
            "batch_processing": self.test_batch_processing(),
            # "stability": self.test_stability(duration_seconds=60),  # Shortened for demo
            "memory_stress": self.test_memory_stress(iterations=20),
        }
        
        # Save results
        output_file = f"/tmp/kvtuner_extended_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"✅ Extended tests completed!")
        print(f"Results saved to: {output_file}")
        print(f"{'='*60}")
        
        return results

if __name__ == "__main__":
    tester = ExtendedTester()
    tester.run_all_extended_tests()
