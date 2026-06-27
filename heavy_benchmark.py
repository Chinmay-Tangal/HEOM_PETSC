import time
import numpy as np
from qutip import sigmax, sigmaz, tensor, qeye, basis
from qutip.solver.heom.bofin_baths import BosonicBath
from qutip.solver.heom.bofin_solvers import HEOMSolver

def run_heavy():
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    if rank == 0:
        print("========================================")
        print("  HEAVY HEOM BENCHMARK (4 Qubit Chain)  ")
        print("========================================")

    # 4 Qubits (16x16 Hamiltonian -> 256x256 Liouvillian)
    sz = sigmaz()
    sx = sigmax()
    I = qeye(2)
    
    # H = sum(sz_i) + nearest-neighbor interaction
    H = (
        tensor(sz, I, I, I) + tensor(I, sz, I, I) + 
        tensor(I, I, sz, I) + tensor(I, I, I, sz) +
        0.5 * (tensor(sx, sx, I, I) + tensor(I, sx, sx, I) + tensor(I, I, sx, sx))
    )
    
    baths = []
    # Add a separate bath to each of the 4 qubits (This massively increases ADOs)
    for i in range(4):
        ops = [I]*4
        ops[i] = sx
        Q = tensor(*ops)
        ck = [1.0]
        vk = [0.1]
        baths.append(BosonicBath(Q, ck, vk, ck, vk))

    # All qubits start in Spin Up
    state0 = tensor(basis(2,0), basis(2,0), basis(2,0), basis(2,0))
    state0 = state0 * state0.dag()
    
    # Short time evolution just to test the step speed
    tlist = np.linspace(0, 1.0, 10)

    # Be careful, depth 8 on 4 baths creates ~3000 ADOs. 
    # Matrix size = 3000 * 256 = 768,000 rows/cols!
    for depth in [4, 6, 8]:
        for backend in ["csr", "petsc"]:
            if rank == 0:
                print(f"\n--- Depth: {depth} | Backend: {backend.upper()} ---")
            
            options = {"backend": backend}
            if backend == "petsc":
                # Give PETSc a fair fight: Adaptive Implicit Solver (BDF)
                options["ts_type"] = "bdf"
                options["ts_adapt"] = "basic"
            
            t0 = time.time()
            solver = HEOMSolver(H, baths, max_depth=depth, options=options)
            t1 = time.time()
            
            if rank == 0:
                print(f"[{backend}] Assembly time: {t1 - t0:.2f} s")
                
            t2 = time.time()
            solver.run(state0, tlist)
            t3 = time.time()

            if rank == 0:
                print(f"[{backend}] Solve time: {t3 - t2:.2f} s")

if __name__ == "__main__":
    run_heavy()
