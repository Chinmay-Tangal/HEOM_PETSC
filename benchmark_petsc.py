import time
import os
import psutil
import numpy as np
from qutip import sigmax, sigmaz, destroy, qeye
from qutip.solver.heom.bofin_baths import BosonicBath
from qutip.solver.heom.bofin_solvers import HEOMSolver

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_benchmark():
    print("="*60)
    print(f"{'Depth':<10} | {'Backend':<10} | {'Assembly (s)':<15} | {'Memory (MB)':<15}")
    print("="*60)

    # We use a simple 2-level system with a Drude-Lorentz bath
    H = sigmaz()
    Q = sigmax()
    ck = [1.0]
    vk = [0.1]
    bath = BosonicBath(Q, ck, vk)

    for depth in [2, 4, 6, 8]:
        for backend in ["csr", "petsc"]:
            try:
                # Memory before
                mem_before = get_memory_mb()
                
                # Assembly time
                t0 = time.time()
                solver = HEOMSolver(H, bath, max_depth=depth, options={"backend": backend})
                t1 = time.time()
                
                # Memory after
                mem_after = get_memory_mb()
                
                assembly_time = t1 - t0
                memory_used = max(0, mem_after - mem_before)
                
                print(f"{depth:<10} | {backend:<10} | {assembly_time:<15.4f} | {memory_used:<15.2f}")
            except ImportError:
                if backend == "petsc":
                    print(f"{depth:<10} | {'petsc':<10} | {'petsc4py missing':<15} | {'N/A':<15}")
                    
    print("="*60)

if __name__ == "__main__":
    try:
        import psutil
    except ImportError:
        print("Please install psutil (pip install psutil) to run this benchmark.")
        exit(1)
        
    run_benchmark()
