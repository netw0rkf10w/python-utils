import numpy as np
import scipy.sparse
import time
import numba

@numba.jit(nopython=True, parallel=True)
def matvec_fast(x, Adata, Aindices, Aindptr, Ashape):
    """
    Fast sparse matrix-vector multiplication
    https://stackoverflow.com/a/47830250/2131200
    """
    numRowsA = Ashape[0]    
    Ax = np.zeros(numRowsA)

    for i in numba.prange(numRowsA):
        Ax_i = 0.0        
        for dataIdx in range(Aindptr[i], Aindptr[i+1]):
            j = Aindices[dataIdx]
            Ax_i += Adata[dataIdx]*x[j]

        Ax[i] = Ax_i
    return Ax


def benchmark(m=10000, n=10000, density=0.1):
    for i in range(5):
        m += 100*i
        n += 100*i
        A = scipy.sparse.random(m, n, density=density, format='csr')
        x = np.random.randn(A.shape[1])

        start = time.time()
        Ax = A.dot(x)
        scipy_time = time.time() - start

        start = time.time()
        AxCheck = matvec_fast(x, A.data, A.indices, A.indptr, A.shape)
        matvec_time = time.time() - start

        print('{}) scipy: {:6f} (s), matvec: {:6f} (s)'.format(i + 1, scipy_time, matvec_time))

        # print('norm =', np.linalg.norm(Ax - AxCheck))

if __name__ == "__main__":
    benchmark(m=10000, n=10000, density=0.01)
