import numpy as np
import scipy.signal


def fconv(x, h):
    """Fast Convolution using FFT
    
    This function convolves two input vectors x and h using Fast Fourier Transform.
    
    Parameters:
        x (array_like): First input vector
        h (array_like): Second input vector (typically the filter/kernel)
    
    Returns:
        ndarray: The convolution result of x and h
        
    Notes:
        This is a Python implementation of the fast convolution algorithm
        originally written by Stephen G. McGovern (2003-2004) in MATLAB.
        For more information about convolution, visit: http://stevem.us/fconv.html
    """
    # Convert inputs to numpy arrays
    x = np.asarray(x)
    h = np.asarray(h)
    
    # Length of the convolution output
    Ly = len(x) + len(h) - 1
    
    # Find smallest power of 2 that is > Ly
    Ly2 = 2**np.ceil(np.log2(Ly)).astype(int)
    
    # Compute FFT of both sequences
    X = np.fft.fft(x, Ly2)
    H = np.fft.fft(h, Ly2)
    
    # Multiply in frequency domain
    Y = X * H
    
    # Inverse FFT
    y = np.real(np.fft.ifft(Y, Ly2))
    
    # Take only the first Ly elements
    y = y[:Ly]
    
    return y 


def adj_fconv(d, r):
    """
    Compute the adjoint Toeplitz product: D.T @ r
    Equivalent to convolution with flipped kernel and cropping

    Args:
        d: 1D convolution kernel (numpy array)
        r: 1D signal (numpy array)

    Returns:
        s: Adjoint convolution result
    """
    d_flipped = np.flip(d)                     # Flip kernel for adjoint
    full_conv = fconv(r, d_flipped)      # Full-mode convolution
    N = len(r) - len(d) + 1
    start = len(d) - 1
    end = start + N
    return full_conv[start:end]


def scipy_stft(y, n_dft) : # stft
    f, t, Y = scipy.signal.stft(y, 
                              window='boxcar',  # rectangular window
                              nperseg=n_dft,    # frame size
                              noverlap=0,       # no overlap
                              nfft=n_dft*2,     # doubled FFT size
                              boundary=None,
                              return_onesided=False,
                              padded=True)      # handle padding
    
    
    Y = np.reshape(Y.T, -1)
    return Y * np.sqrt(n_dft/2)


def scipy_istft(Y, n_dft) :
    Y = Y.reshape((-1, n_dft*2)).T
    _ , y = scipy.signal.istft(Y, 
                              window='boxcar',  # rectangular window
                              nperseg=n_dft,    # frame size
                              noverlap=0,       # no overlap
                              nfft=n_dft*2,     # doubled FFT size
                              input_onesided=False,
                              boundary=None)      # handle padding
    return np.real(y) /np.sqrt(n_dft/2)