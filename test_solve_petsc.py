import time
import numpy as np
from qutip import sigmax, sigmaz, basis
from qutip.solver.heom.bofin_baths import BosonicBath
from qutip.solver.heom.bofin_solvers import HEOMSolver

def run_solver():
    # Setup MPI so we only print from Rank 0
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    if rank == 0:
        print("Initializing HEOM system for Time Evolution...")

    # Define physics
    H = sigmaz()
    Q = sigmax()
    ck = [1.0]
    vk = [0.1]
    bath = BosonicBath(Q, ck, vk, ck, vk)
    depth = 4

    # Initial system state (spin up)
    state0 = basis(2, 0) * basis(2, 0).dag()

    # Time steps for evolution
    tlist = np.linspace(0, 5, 50)

    for backend in ["csr", "petsc"]:
        if rank == 0:
            print(f"\n--- Testing {backend.upper()} backend ---")
            
        # 1. Assemble the matrix
        t0 = time.time()
        solver = HEOMSolver(H, bath, max_depth=depth, options={"backend": backend})
        t1 = time.time()
        
        if rank == 0:
            print(f"[{backend}] Assembly time: {t1 - t0:.4f} s")
            print(f"[{backend}] Starting time evolution...")

        # 2. Solve the ODE over time
        t2 = time.time()
        result = solver.run(state0, tlist, e_ops=[sigmaz()])
        t3 = time.time()

        # 3. Print the results
        if rank == 0:
            print(f"[{backend}] Solve time: {t3 - t2:.4f} s")
            print(f"[{backend}] Final Expectation Value of sigma_z: {result.expect[0][-1]:.4f}")

if __name__ == "__main__":
    run_solver()
