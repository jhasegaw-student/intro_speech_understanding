import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.

    @param:
    f (scalar): frequency of the root tone, in Hertz
    Fs (scalar): sampling frequency, in samples/second

    @return:
    x (array): a one-half-second waveform containing the chord
    
    A major chord is three notes, played at the same time:
    (1) The root tone (f)
    (2) A major third, i.e., four semitones above f
    (3) A major fifth, i.e., seven semitones above f
    '''
    #raise RuntimeError("You need to write this part")
    n = np.arange(int(Fs/2))
    omega0 = 2*np.pi*f/Fs
    G = np.power(2,1/12)
    note1 = np.cos(omega0 * n)   # e.g., A in an A-major chord
    note2 = np.cos(omega0 * (G**4) * n)   # e.g., C# in an A-major chord
    note3 = np.cos(omega0 * (G**7) * n)   # e.g., E in an A-major chord
    return note1+note2+note3

def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    
    @param:
    N (scalar): number of columns in the transform matrix
    
    @result:
    W (NxN array): a matrix of dtype='complex' whose (k,n)^th element is:
           W[k,n] = cos(2*np.pi*k*n/N) - j*sin(2*np.pi*k*n/N)
    '''
    #raise RuntimeError("You need to write this part")
    W = np.zeros((N,N), dtype='complex')
    for k in range(N):
        for n in range(N):
            W[k,n] = np.cos(2*np.pi*k*n/N) - (0+1j)*np.sin(2*np.pi*k*n/N)
    return W

def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.

    @param:
    x (array): the waveform
    Fs (scalar): sampling frequency (samples/second)

    @return:
    f1, f2, f3: The three loudest frequencies (in Hertz)
      These should be sorted so f1 < f2 < f3.
    '''
    #raise RuntimeError("You need to write this part")
    N = len(x)
    X = np.fft.fft(x)
    MX = np.abs(X[:int(N/2)])
    k1 = np.argmax(MX)
    MX[k1] = 0
    k2 = np.argmax(MX)
    MX[k2] = 0
    k3 = np.argmax(MX)
    f1, f2, f3 = (k1/N)*Fs, (k2/N)*Fs, (k3/N)*Fs
    return sorted((f1, f2, f3))
    
