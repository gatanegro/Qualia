import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift

def mirror_operator(phi):
    phi_fft = fft(phi)  # Fourier transform
    phi_fft_conj = np.conj(phi_fft)  # Complex conjugate
    phi_fft_shifted = fftshift(phi_fft_conj)  # Invert frequencies
    hilbert = -1j * np.sign(np.arange(-128, 128))  # Hilbert kernel
    qualia_fft = phi_fft_shifted * hilbert
    qualia = ifft(qualia_fft).real  # Inverse transform
    return qualia

# Gaussian field
x = np.linspace(-10, 10, 256)
phi = np.exp(-x**2)  # Initial field
qualia = mirror_operator(phi)
phi_dark = phi - qualia

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x, phi, label="Original Field $\phi(x)$")
plt.plot(x, qualia, label="Qualia Field $I[\phi](x)$")
plt.plot(x, phi_dark, label="Dark Field $\phi_{\mathrm{dark}}(x)$")
plt.title("Mirror Operator Transform")
plt.legend()
plt.savefig("qualia_simulation.png", dpi=300)
plt.show()


