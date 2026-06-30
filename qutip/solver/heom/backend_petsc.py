import numpy as np

class PETScGatherHEOMRHS:
    """ A class for collecting elements of the right-hand side matrix
        of the HEOM and streaming them directly into a distributed PETSc Matrix
        to avoid Python object memory overhead.
    """
    def __init__(self, f_idx, block, nhe):
        self._block_size = block
        self._n_blocks = nhe
        self._f_idx = f_idx
        
        try:
            from petsc4py import PETSc
        except ImportError:
            raise ImportError("petsc4py is required for the PETSc backend.")
            
        comm = PETSc.COMM_WORLD
        size = comm.getSize()
        rank = comm.getRank()
        
        global_size = block * nhe
        n_local_blocks = nhe // size
        remainder = nhe % size
        
        if rank < remainder:
            local_blocks = n_local_blocks + 1
        else:
            local_blocks = n_local_blocks
            
        local_size = local_blocks * block
        
        self.mat = PETSc.Mat().create(comm)
        self.mat.setSizes(((local_size, global_size), (local_size, global_size)))
        self.mat.setType(PETSc.Mat.Type.MPIAIJ)
        
        # Preallocation estimate
        max_connections = 50 * block
        self.mat.setPreallocationNNZ((max_connections, max_connections))
        self.mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

    def add_op(self, row_he, col_he, op):
        from petsc4py import PETSc
        row_blk = self._f_idx(row_he)
        col_blk = self._f_idx(col_he)
        
        row_indices = np.arange(row_blk * self._block_size, (row_blk + 1) * self._block_size, dtype=np.int32)
        col_indices = np.arange(col_blk * self._block_size, (col_blk + 1) * self._block_size, dtype=np.int32)
        
        self.mat.setValues(row_indices, col_indices, op.as_scipy().todense(), addv=PETSc.InsertMode.ADD_VALUES)

    def gather(self, L_sys=None):
        from petsc4py import PETSc
        if L_sys is not None and L_sys.isconstant:
            L_sys_dense = L_sys(0).data.as_scipy().todense()
            
            comm = PETSc.COMM_WORLD
            size = comm.getSize()
            rank = comm.getRank()
            
            n_local_blocks = self._n_blocks // size
            remainder = self._n_blocks % size
            if rank < remainder:
                start_block = rank * (n_local_blocks + 1)
                end_block = start_block + n_local_blocks + 1
            else:
                start_block = rank * n_local_blocks + remainder
                end_block = start_block + n_local_blocks
            
            for r_blk in range(start_block, end_block):
                row_indices = np.arange(r_blk * self._block_size, (r_blk + 1) * self._block_size, dtype=np.int32)
                self.mat.setValues(row_indices, row_indices, L_sys_dense, addv=PETSc.InsertMode.ADD_VALUES)
                
        self.mat.assemblyBegin()
        self.mat.assemblyEnd()
        return self.mat
