import numpy as np


def to_db(x, ref=1.0, eps=1e-10):
    """Convert amplitude to decibel.
    
    Args:
        x: Input signal
        ref: Reference value (default: 1.0)
        eps: Small value to prevent log of zero (default: 1e-10)
        
    Returns:
        Signal in decibels
    """
    return 20 * np.log10(np.abs(x) + eps) - 20 * np.log10(ref)

def get_edc(h: np.ndarray, log_edc: bool = True) -> np.ndarray:
    assert np.ndim(h) == 1
    edc = np.flip(np.cumsum(np.flip(np.abs(h) ** 2)))
    if log_edc:
        return 10 * np.log10(edc + 1e-10)
    else:
        return edc