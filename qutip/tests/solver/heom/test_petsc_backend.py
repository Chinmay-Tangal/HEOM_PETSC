import pytest
import numpy as np

from qutip import QobjEvo, Qobj, sigmax, sigmaz, destroy, qeye
from qutip.solver.heom.bofin_baths import BosonicBath, FermionicBath
from qutip.solver.heom.bofin_solvers import HEOMSolver

# Try to import petsc4py
try:
    from petsc4py import PETSc
    has_petsc = True
except ImportError:
    has_petsc = False

@pytest.mark.skipif(not has_petsc, reason="petsc4py not available")
class TestPETScBackend:
    def test_bosonic_bath(self):
        """Test PETSc matrix assembly for a Bosonic bath"""
        H = sigmaz()
        Q = sigmax()
        
        # Define a simple Drude-Lorentz bath (bosonic)
        ck = [1.0 + 0.1j, 0.5 - 0.2j]
        vk = [0.1, 1.0]
        bath = BosonicBath(Q, ck, vk, ck, vk)
        
        solver_csr = HEOMSolver(H, bath, max_depth=2, options={"backend": "csr"})
        solver_petsc = HEOMSolver(H, bath, max_depth=2, options={"backend": "petsc"})
        
        self._compare_matrices(solver_csr, solver_petsc)

    def test_fermionic_bath(self):
        """Test PETSc matrix assembly for a Fermionic bath"""
        H = sigmaz()
        Q = destroy(2)
        
        ck = [1.0, 0.5]
        vk = [0.1, 1.0]
        bath = FermionicBath(Q, ck, vk, ck, vk)
        
        solver_csr = HEOMSolver(H, bath, max_depth=2, options={"backend": "csr"})
        solver_petsc = HEOMSolver(H, bath, max_depth=2, options={"backend": "petsc"})
        
        self._compare_matrices(solver_csr, solver_petsc)

    def test_multiple_mixed_baths(self):
        """Test PETSc matrix assembly for multiple mixed baths"""
        H = sigmaz()
        Q_b = sigmax()
        Q_f = destroy(2)
        
        # Bosonic bath
        ck_b = [1.0]
        vk_b = [0.1]
        bath_b = BosonicBath(Q_b, ck_b, vk_b, ck_b, vk_b)
        
        # Fermionic bath
        ck_f = [0.5]
        vk_f = [1.0]
        bath_f = FermionicBath(Q_f, ck_f, vk_f, ck_f, vk_f)
        
        solver_csr = HEOMSolver(H, [bath_b, bath_f], max_depth=2, options={"backend": "csr"})
        solver_petsc = HEOMSolver(H, [bath_b, bath_f], max_depth=2, options={"backend": "petsc"})
        
        self._compare_matrices(solver_csr, solver_petsc)

    def test_large_hierarchy(self):
        """Test PETSc matrix assembly for a larger hierarchy depth and dimension"""
        H = sigmaz()
        Q = sigmax()
        
        # Define a simple Drude-Lorentz bath (bosonic)
        ck = [1.0]
        vk = [0.1]
        bath = BosonicBath(Q, ck, vk, ck, vk)
        
        solver_csr = HEOMSolver(H, bath, max_depth=5, options={"backend": "csr"})
        solver_petsc = HEOMSolver(H, bath, max_depth=5, options={"backend": "petsc"})
        
        self._compare_matrices(solver_csr, solver_petsc)

    def test_time_dependent_raises(self):
        """Test that time-dependent Liouvillians raise NotImplementedError"""
        H = [sigmaz(), [sigmax(), "sin(t)"]]
        Q = sigmax()
        
        ck = [1.0]
        vk = [0.1]
        bath = BosonicBath(Q, ck, vk, ck, vk)
        
        with pytest.raises(NotImplementedError, match="only supports time-independent"):
            # The error should be raised when _calculate_rhs is triggered,
            # which happens during solver initialization
            HEOMSolver(H, bath, max_depth=2, options={"backend": "petsc"})

    def _compare_matrices(self, solver_csr, solver_petsc):
        """Helper to compare QuTiP CSR matrix and PETSc assembled matrix"""
        # Ensure hierarchy ordering and ADO ordering are identical
        assert solver_csr.ados.labels == solver_petsc.ados.labels
        
        # Extract SciPy sparse matrix from QuTiP's RHS
        if isinstance(solver_csr.rhs, QobjEvo):
            csr_scipy = solver_csr.rhs(0).data.as_scipy()
        else:
            csr_scipy = solver_csr.rhs.data.as_scipy()
            
        # Extract SciPy sparse matrix from PETSc Mat
        petsc_mat = solver_petsc.rhs
        
        # Check dimensions
        assert csr_scipy.shape == petsc_mat.getSize()
        
        if petsc_mat.getSize() != petsc_mat.getLocalSize():
            pytest.skip("MPI matrix comparison requires gathering to rank 0, skipping for now.")
        
        # petsc_mat.getValuesCSR() returns (indptr, indices, data)
        indptr, indices, data = petsc_mat.getValuesCSR()
        
        # Check NNZ
        assert csr_scipy.nnz == len(data)
        
        # Check sparsity pattern exactly
        np.testing.assert_array_equal(csr_scipy.indptr, indptr)
        np.testing.assert_array_equal(csr_scipy.indices, indices)
        
        # Check nonzero values
        np.testing.assert_allclose(csr_scipy.data, data, rtol=1e-12, atol=1e-14)
