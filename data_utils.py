import scipy.signal
import librosa
import numpy as np


def load_excitation_signal(audio_filepath, fs, start_s, end_s): 
    # Load real audio file
    x, _ = librosa.load(audio_filepath, sr=fs, mono=True)
    x = x[fs*start_s:fs*end_s]

    return x 


def load_synthetic_impulse_response(decay_db, fs, filter_length): 
    gamma = -decay_db/fs
    # Generate synthetic impulse response
    h = np.random.randn(filter_length) * 10. ** (gamma * np.arange(filter_length) / 20.)
    h = h / np.linalg.norm(h, 2)
    return h 


def load_real_impulse_response(audio_filepath, fs): 
    h, _ = librosa.load(audio_filepath, sr=fs, mono=True)
    h = h / np.linalg.norm(h, 2)
    return h   


def add_bg_noise_with_snr(x, snr_db):
    signal_power = np.mean(x**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(*x.shape)
    return x + noise, noise


def add_real_nonstationary_noise(x, filepath, fs, duration, loc_s, amplitude_multiplier) : 
    noise, _ = librosa.load(filepath, sr=fs, mono=True)
    if duration is not None : 
        noise = noise[:int(duration*fs)]

    # Create a zero array for the noise, same length as x
    noise_full = np.zeros_like(x)

    # Calculate the maximum length the noise can be without exceeding x
    max_noise_len = len(x) - loc_s*fs
    cropped_noise = noise[:max_noise_len]

    # Insert the cropped noise at loc_s
    noise_full[loc_s*fs:loc_s*fs+len(cropped_noise)] = cropped_noise * amplitude_multiplier

    # Add the noise to the signal
    x_noisy = x + noise_full

    return x_noisy, noise_full


def add_synthetic_nonstationary_noise(x, n_peaks): 
    n_peaks = 10
    peakIndex = np.random.permutation(np.arange(len(x)))[:n_peaks]
    noise = np.zeros(len(x))
    noise[peakIndex] = 10 * np.random.randn(n_peaks)
    x_noisy = x + noise 
    return  x_noisy, noise


# def add_high_frequency_noise(x, cutoff_fs, fs):
#     # Add high frequency noise above 14kHz
#     # Create noise signal same length as audio
#     noise = np.random.randn(len(x))

#     # Create high-pass filter above 14kHz
#     nyquist = fs/2
#     cutoff = cutoff_fs / nyquist
#     b, a = scipy.signal.butter(8, cutoff, btype='high')

#     # Filter noise and scale it down
#     filtered_noise = scipy.signal.filtfilt(b, a, noise)

#     # Add filtered noise to signal
#     x = x + filtered_noise / 100.0 
#     return x

def add_high_frequency_noise(x, cutoff_fs, fs, snr_db=40):
    """
    Add high-frequency noise above cutoff_hz at a given SNR.
    
    Args:
        x        : input signal
        cutoff_hz: cutoff frequency for high-pass noise (Hz)
        fs       : sampling rate
        snr_db   : desired SNR in dB (default: 20 dB)
    Returns:
        x_noisy  : signal + high-frequency noise
    """
    # Generate white noise
    noise = np.random.randn(len(x))

    # High-pass filter above cutoff_hz
    nyquist = fs / 2
    cutoff = cutoff_fs / nyquist
    b, a = scipy.signal.butter(8, cutoff, btype='high')
    noise = scipy.signal.filtfilt(b, a, noise)

    # Match noise power to target SNR
    Px = np.mean(x**2)
    Pn = np.mean(noise**2)
    target_Pn = Px / (10**(snr_db / 10))
    noise = noise * np.sqrt(target_Pn / (Pn + 1e-12))

    # Add to signal
    x_noisy = x + noise
    return x_noisy

def add_silence(x, fs, silence_duration_s):
    silence = np.zeros(int(silence_duration_s * fs))
    return np.concatenate([x, silence])


def get_equalised_filter_lpc(x, order=200):
    """LPC-based approach - often more stable for spectral flattening"""
    # Get LPC coefficients
    a = librosa.lpc(x, order=order)
    
    # Normalize
    a = a / np.sum(np.abs(a))
    
    return a


def equalise_with_filter(x, lpc_coeffs):
    return scipy.signal.lfilter(lpc_coeffs, [1], x)
