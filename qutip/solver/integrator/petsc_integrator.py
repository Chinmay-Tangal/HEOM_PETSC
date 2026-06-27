import numpy as np
from qutip.core import data as _data
from qutip.solver.integrator.integrator import Integrator

class IntegratorPETSc(Integrator):
    """
    ODE Integrator that uses petsc4py TS (Time Stepping) solver.
    This integrator handles PETSc.Mat directly instead of QobjEvo.
    """
    integrator_options = {
        "ts_type": "rk",     # Runge-Kutta by default
        "ts_adapt": "none",  # Adaptivity: 'none', 'basic', etc.
        "dt": 1e-4,          # Initial time step
        "max_steps": 10000,
        "atol": 1e-8,
        "rtol": 1e-6,
    }
    
    support_time_dependant = False
    supports_blackbox = False
    name = "petsc"
    method = "ts"

    def _prepare(self):
        try:
            from petsc4py import PETSc
        except ImportError:
            raise ImportError("petsc4py is required to use IntegratorPETSc.")
            
        self.PETSc = PETSc
        # self.system is the PETSc.Mat from backend_petsc.py
        self.mat = self.system
        
        self.ts = PETSc.TS().create()
        self.ts.setProblemType(PETSc.TS.ProblemType.LINEAR)
        self.ts.setEquationType(PETSc.TS.EquationType.ODE_EXPLICIT)
        
        self.ts.setRHSFunction(PETSc.TS.computeRHSFunctionLinear)
        self.ts.setRHSJacobian(PETSc.TS.computeRHSJacobianConstant, self.mat, self.mat)
        
        self.ts.setType(self.options.get("ts_type", "rk"))
        self.ts.setTimeStep(self.options.get("dt", 1e-4))
        self.ts.setMaxSteps(self.options.get("max_steps", 10000))
        self.ts.setTolerances(
            atol=self.options.get("atol", 1e-8),
            rtol=self.options.get("rtol", 1e-6)
        )
        
        # We need a PETSc Vec for the state
        rstart, rend = self.mat.getOwnershipRange()
        self.vec = self.mat.createVecRight()
        self.vec.setFromOptions()
        
        self.ts.setUp()
        self.name = f"petsc_ts_{self.options.get('ts_type', 'rk')}"

    def set_state(self, t, state0):
        # state0 is a qutip.Data object (usually Dense), we need to extract its values
        # state0 represents the full hierarchy state
        state_np = state0.to_array().flatten()
        
        rstart, rend = self.mat.getOwnershipRange()
        local_state = state_np[rstart:rend]
        
        self.vec.setValues(range(rstart, rend), local_state)
        self.vec.assemblyBegin()
        self.vec.assemblyEnd()
        
        self.ts.setTime(t)
        self.ts.setSolution(self.vec)
        self._is_set = True

    def integrate(self, t, copy=True):
        if not self._is_set:
            raise RuntimeError("The initial state must be set using set_state before integrating.")
            
        self.ts.setMaxTime(t)
        self.ts.setExactFinalTime(self.PETSc.TS.ExactFinalTime.MATCHSTEP)
        self.ts.solve(self.vec)
        
        # After solve, we need to gather the distributed vector back to a QuTiP Data object
        # Since QuTiP's outer API expects a full Qobj state, we must gather the vector to rank 0 (or all ranks).
        # We will use VecScatter to gather to all ranks so QuTiP works seamlessly.
        scatter, vec_seq = self.PETSc.Scatter.toAll(self.vec)
        scatter.scatter(self.vec, vec_seq, self.PETSc.InsertMode.INSERT_VALUES, self.PETSc.ScatterMode.FORWARD)
        
        gathered_np = vec_seq.getArray().copy()
        
        # Convert back to qutip.Data Dense
        shape = (self.mat.getSize()[1], 1)
        state_data = _data.Dense(gathered_np.reshape(shape))
        
        current_t = self.ts.getTime()
        return current_t, state_data

    def mcstep(self, t, copy=True):
        raise NotImplementedError("Monte Carlo steps are not supported for PETSc integrator.")
