import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from typing import List, Dict, Any
from scipy.signal import find_peaks
from parameters import generate_parameters
from qd import generate_quantum_dot_spectrum
from vary_parameters import generate_parameter_variators

st.title("Quantum Dot Spectrum Simulation")

(
    energies,
    exciton_energy, 
    exciton_linewidth, 
    fine_structure_splitting,
    power,
    polarizer_angle_deg, 
    noise,
    prominence
) = generate_parameter_variators()

parameters = generate_parameters(
    power,
    polarizer_angle_deg,
    energies,
    exciton_energy, 
    exciton_linewidth, 
    fine_structure_splitting
)

spectrum = generate_quantum_dot_spectrum(
    energies=energies,
    particles=parameters['particles'],
    polarizer_angle_rad=parameters['polarizer_angle_deg'] * math.pi / 180,
    power=parameters['power'],
    noise=noise
)

st.sidebar.header("Peak Detection Parameters")

peaks, properties = find_peaks(spectrum, prominence=prominence, distance=20)

peak_energies = energies[peaks]
peak_intensities = spectrum[peaks]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(energies, spectrum, label='Spectrum', color='blue')
ax.set_xlabel('Energy')
ax.set_ylabel('Intensity')
ax.set_title('Quantum Dot Spectrum with Detected Peaks')

ax.plot(peak_energies, peak_intensities, "x", label='Detected Peaks', color='red', markersize=10, markeredgewidth=2)

for energy, intensity in zip(peak_energies, peak_intensities):
    ax.annotate(f"{energy:.2f}", xy=(energy, intensity), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)

ax.legend()
ax.grid(True)

st.pyplot(fig)

st.subheader("Particles")
st.text(f"All peak energies are calculated relative to the exciton line.")
st.text(f"FSS is set by hand.")
st.text(f"Power dependence is empirically determined.")
st.dataframe(parameters['particles'])

st.subheader("Detected Peaks Information")

if len(peaks) > 0:
    peak_data = pd.DataFrame({
        'Peak Energy': peak_energies,
        'Intensity': peak_intensities
    })
    st.dataframe(peak_data)
else:
    st.write("No peaks detected with the current parameters.")
