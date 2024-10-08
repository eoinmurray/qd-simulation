import numpy as np
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
# -----------------------------
# 2. Define Functions
# -----------------------------

def lorentzian(energies: np.ndarray, center_energy: float, linewidth: float) -> np.ndarray:
    """
    Lorentzian function: Computes the Lorentzian line shape for modeling the spectrum of a quantum dot.

    Parameters:
    - energies (np.ndarray): Array of energy values at which to evaluate the Lorentzian.
    - center_energy (float): Central energy value of the Lorentzian peak (E₀).
    - linewidth (float): Full width at half maximum (FWHM) of the Lorentzian peak (Γ).

    Returns:
    - np.ndarray: Lorentzian values at the given energy array.
    """
    half_linewidth = linewidth / 2  # Half-width at half-maximum (HWHM)
    numerator = half_linewidth ** 2
    denominator = (np.array(energies) - center_energy) ** 2 + half_linewidth ** 2
    return numerator / denominator


def generate_quantum_dot_spectrum(
    energies: np.ndarray,
    particles: List[Dict[str, Any]],
    polarizer_angle_rad,
    power,
) -> List[Dict[str, Any]]:
    """
    Spectrum generation function: Models the spectrum of an arbitrary number of excitonic particles with polarization effects.

    Parameters:
    - energies (np.ndarray): Array of energy values over which to evaluate the spectrum (meV).
    - particles (List[Dict[str, Any]]): List of particle objects, each with properties:
        - name (str): Name of the particle (e.g., 'Exciton', 'Biexciton').
        - energy (float): Energy at which the particle peak occurs (meV).
        - linewidth (float): Linewidth of the peak (meV).
        - fss (float): Fine structure splitting (ΔE), can be zero (meV).
        - intensityScaling (float): Scaling factor for intensity (e.g., α, β).
        - power_dependence (float, optional): Exponent indicating how the intensity scales with power (default is 1 for linear).
    - polarizer_angle_rad (float): Angle of polarization in radians.
    - power

    Returns:
    - List[Dict[str, Any]]: List of dictionaries containing totalIntensity and individual particle intensities for each power level.
    """

    # Precompute cosine and sine squared of the polarizer angle for efficiency
    cos_squared_angle = np.cos(polarizer_angle_rad) ** 2
    sin_squared_angle = np.sin(polarizer_angle_rad) ** 2

    # Initialize total intensity array
    total_intensity = np.zeros_like(energies)

    # Initialize dictionary to hold individual particle intensities
    particle_intensities = {}

    # Loop over each excitonic particle
    for particle in particles:
        # Initialize particle intensity array
        particle_intensity = np.zeros_like(energies)

        # Compute the scaling factor for this particle based on power level
        intensity_scaling = particle['intensity_scaling'] * (power ** particle.get('power_dependence', 1))

        # Determine polarization factors based on fine structure splitting
        fss = particle.get('fss', 0.0)
        if not fss:
            # No fine structure splitting
            polarization_factors = [{'factor': 1.0, 'energy_shift': 0.0}]
        else:
            polarization_factors = [
                {'factor': cos_squared_angle, 'energy_shift': fss / 2},
                {'factor': sin_squared_angle, 'energy_shift': -fss / 2},
            ]

        # Loop over each polarization component of the particle
        for polarization in polarization_factors:
            shifted_energy = particle['energy'] + polarization['energy_shift']
            factor = polarization['factor'] * intensity_scaling

            # Compute the Lorentzian for this component and accumulate intensity
            lorentz_values = lorentzian(energies, shifted_energy, particle['linewidth'])
            intensity = factor * lorentz_values

            # Accumulate intensities
            particle_intensity += intensity
            total_intensity += intensity

        # Store individual particle intensity
        particle_intensities[particle['name']] = particle_intensity.tolist()
    
    return total_intensity


if __name__ == "__main__":
    # Define parameters for generating quantum dot spectrum
    energies = np.linspace(1580, 1600, 1000)
    particles = [
        {
            'name': 'Exciton',
            'energy': 1588.0,
            'linewidth': 0.01,
            'fss': 0.0,
            'intensity_scaling': 1.0,
            'power_dependence': 3.0
        },
        {
            'name': 'Biexciton',
            'energy': 1590.0,
            'linewidth': 0.02,
            'fss': 0.0,
            'intensity_scaling': 0.5,
            'power_dependence': 4.0
        }
    ]
    polarizer_angle_rad = 45 * np.pi / 180
    power = 3.0

    # Generate quantum dot spectrum
    spectrum = generate_quantum_dot_spectrum(energies, particles, polarizer_angle_rad, power)

    # Plot the generated spectrum
    plt.figure(figsize=(12, 4))
    plt.plot(energies, spectrum, label='Quantum Dot Spectrum')
    plt.xlabel('Energy (meV)')
    plt.ylabel('Intensity')
    plt.title('Quantum Dot Spectrum')
    plt.legend()
    plt.grid(True)
    plt.show()