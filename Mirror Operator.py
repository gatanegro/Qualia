import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift

# Define Mirror Operator I[ϕ]


def mirror_operator(phi):
    phi_fft = fftn(phi)  # Fourier transform
    phi_fft_conj = np.conj(phi_fft)  # Complex conjugate
    phi_fft_shifted = fftshift(phi_fft_conj)  # Invert k,ω
    hilbert = -1j * np.sign(phi_fft_shifted)  # Hilbert transform
    qualia_fft = phi_fft_shifted * hilbert
    qualia = ifftn(qualia_fft).real  # Inverse transform
    return qualia


# Example: Gaussian energy field
x = np.linspace(-10, 10, 256)
phi = np.exp(-x**2)  # Simple 1D field
qualia = mirror_operator(phi)

# Plot
plt.plot(x, phi, label="Original Field ϕ(x)")
plt.plot(x, qualia, label="Qualia Field I[ϕ](x)")
plt.title("Mirror Operator Transform")
plt.legend()
plt.show()
