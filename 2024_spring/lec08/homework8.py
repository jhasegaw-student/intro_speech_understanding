import numpy as np

def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    
    @param:
    N (scalar): number of columns in the transform matrix
    
    @result:
    W (NxN array): a matrix of dtype='complex' whose (k,n)^th element is:
           W[k,n] = cos(2*np.pi*k*n/N) - j*sin(2*np.pi*k*n/N)
    '''
    W = np.zeros((N,N), dtype=complex)
    j = 0+1j
    for k in range(N):
        for n in range(N):
            W[k,n] = np.cos(2*np.pi*k*n/N) - j*np.sin(2*np.pi*k*n/N)
    return W
    