"""Monitor GPU usage during training"""

import subprocess
import time
import os

try:
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 80)
        print("GPU UTILIZATION MONITOR")
        print("=" * 80)
        print()
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                data = result.stdout.strip().split(', ')
                if len(data) >= 6:
                    gpu_name, gpu_util, mem_util, mem_used, mem_total, temp = data
                    
                    print(f"GPU: {gpu_name}")
                    print(f"GPU Utilization: {gpu_util}%")
                    print(f"Memory Utilization: {mem_util}%")
                    print(f"Memory Used: {mem_used} MB / {mem_total} MB")
                    print(f"Temperature: {temp}°C")
                    
                    # Visual bars
                    gpu_bar = '█' * int(float(gpu_util) / 5) + '░' * (20 - int(float(gpu_util) / 5))
                    mem_bar = '█' * int(float(mem_util) / 5) + '░' * (20 - int(float(mem_util) / 5))
                    
                    print(f"\nGPU:  [{gpu_bar}] {gpu_util}%")
                    print(f"VRAM: [{mem_bar}] {mem_util}%")
            else:
                print("⚠ nvidia-smi not available")
                
        except subprocess.TimeoutExpired:
            print("⚠ nvidia-smi timeout")
        except FileNotFoundError:
            print("⚠ nvidia-smi not found")
            break
        
        print(f"\n{'='*80}")
        print("Press Ctrl+C to stop monitoring")
        time.sleep(2)
        
except KeyboardInterrupt:
    print("\n\n✓ Monitoring stopped")
