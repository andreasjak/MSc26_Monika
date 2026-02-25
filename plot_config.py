import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
import scipy.signal
import numpy as np

# Figure sizing (in inches)
text_width = 7.1413       # Full LaTeX text width
column_width = 3.48761    # One-column width

# Font setup
font = {
    'family': 'serif',
    'serif': ['Times New Roman'],
    'size': 9
}

# PGF + LaTeX rendering settings
def configure_plots():
    plt.rcParams.update({
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "font.family": "serif",
        "font.size": 9,
        "text.latex.preamble": r"""
            \usepackage{siunitx}
            \sisetup{detect-all}
            \usepackage{helvet}
            \usepackage{sansmath}
            \sansmath
        """
    })

    # Apply font
    mpl.rc('font', **font)

    # Optional: PGF backend for LaTeX integration (uncomment to use)
    # mpl.use("pgf")

    # Use SciencePlots style if installed
    try:
        plt.style.use(["science", "grid"])
    except:
        print("Warning: 'science' style not found. Install with `pip install SciencePlots`")


def scipy_stft_plot(y, n_dft, fs) : 
    f, t, Y = scipy.signal.stft(y, 
                              window='boxcar',  # rectangular window
                              nperseg=n_dft,    # frame size
                              noverlap=0,       # no overlap
                              nfft=n_dft*2,     # doubled FFT size
                              boundary=None,
                              return_onesided=True,
                              fs = fs,
                              padded=True)      # handle padding
    
    
    return f,t,np.abs(Y)

# Useful sizes
TEXT_WIDTH = text_width
COLUMN_WIDTH = column_width 