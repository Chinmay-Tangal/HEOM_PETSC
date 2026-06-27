import numpy as np

def gather_heom_rhs_petsc(ops_list, block_size, n_blocks, L_sys=None):
    """
    Assemble the HEOM Liouvillian using PETSc.
    
    Parameters
    ----------
    ops_list : list of tuples
        List of (row_idx, col_idx, op) where op is a CSR matrix block.
    block_size : int
        The size of a single ADO block (e.g. n^2 for system dimension n).
    n_blocks : int
        The number of ADOs in the hierarchy.
    L_sys : QobjEvo, optional
        The time-independent system Liouvillian to add to the diagonal blocks.
    
    Returns
    -------
    mat : PETSc.Mat
        The assembled PETSc sparse matrix.
    """
    try:
        from petsc4py import PETSc
    except ImportError:
        raise ImportError("petsc4py is required for the PETSc backend.")

    if np.dtype(PETSc.ScalarType) != np.complex128:
        raise ValueError("petsc4py must be configured with complex scalars (PETSc.ScalarType == np.complex128) for QuTiP HEOM.")

    global_size = block_size * n_blocks
    
    # Pre-extract scipy CSR matrices to avoid repeated python allocations
    # and duplicate work
    blocks = []
    for r_blk, c_blk, op in ops_list:
        csr = op.as_scipy()
        blocks.append((r_blk, c_blk, csr.indptr, csr.indices, csr.data))
        
    if L_sys is not None and L_sys.isconstant:
        l_sys_csr = L_sys(0).to("csr").data.as_scipy()
        # Add L_sys to every diagonal block
        for r_blk in range(n_blocks):
            blocks.append((r_blk, r_blk, l_sys_csr.indptr, l_sys_csr.indices, l_sys_csr.data))

    # 1. Create PETSc Mat and determine MPI ownership
    mat = PETSc.Mat().create()
    mat.setSizes([global_size, global_size])
    mat.setType("aij")
    mat.setUp()
    
    rstart, rend = mat.getOwnershipRange()
    local_rows = rend - rstart
    
    # 2. Compute proper MPI preallocation (d_nnz, o_nnz)
    d_nnz = np.zeros(local_rows, dtype=np.int32)
    o_nnz = np.zeros(local_rows, dtype=np.int32)
    
    for r_blk, c_blk, indptr, indices, data in blocks:
        row_start = r_blk * block_size
        row_end = row_start + block_size
        
        overlap_start = max(row_start, rstart)
        overlap_end = min(row_end, rend)
        
        if overlap_start < overlap_end:
            col_start = c_blk * block_size
            cols = col_start + indices
            
            for i in range(overlap_start - row_start, overlap_end - row_start):
                start, end = indptr[i], indptr[i+1]
                if start < end:
                    local_row_idx = (row_start + i) - rstart
                    row_cols = cols[start:end]
                    
                    diag_count = np.count_nonzero((row_cols >= rstart) & (row_cols < rend))
                    d_nnz[local_row_idx] += diag_count
                    o_nnz[local_row_idx] += (end - start) - diag_count

    mat.setPreallocationNNZ((d_nnz, o_nnz))
    
    # 3. Assemble local blocks
    for r_blk, c_blk, indptr, indices, data in blocks:
        row_start = r_blk * block_size
        row_end = row_start + block_size
        
        overlap_start = max(row_start, rstart)
        overlap_end = min(row_end, rend)
        
        if overlap_start < overlap_end:
            col_start = c_blk * block_size
            
            for i in range(overlap_start - row_start, overlap_end - row_start):
                start = indptr[i]
                end = indptr[i+1]
                if start < end:
                    global_row = row_start + i
                    global_cols = col_start + indices[start:end]
                    vals = data[start:end]
                    mat.setValues(global_row, global_cols, vals, addv=PETSc.InsertMode.ADD_VALUES)
                    
    mat.assemblyBegin()
    mat.assemblyEnd()
    
    return mat
