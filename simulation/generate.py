import matplotlib.pyplot as plt
import numpy as np
import math
from qd import generate_quantum_dot_spectrum
from vary_parameters import generate_parameter_variators
from parameters import generate_parameters

def generate_data(num_samples, seq_length):
    data = np.zeros((num_samples, seq_length))
    for i in range(num_samples):
        series = np.zeros(seq_length)
        # Randomly insert sharp peaks
        num_peaks = np.random.randint(1, 5)
        for _ in range(num_peaks):
            peak_position = np.random.randint(0, seq_length)
            peak_height = np.random.uniform(1, 5)
            series[peak_position] = peak_height
            # Normalize data but keep peak values prominent
            max_value = series.max()  # Use the 99th percentile to avoid extreme peaks
            series = series / max_value
            series = np.clip(series, 0, 1)
        data[i] = series
        
    return data

def generate_qd_data(num_samples, seq_length, normalization=True):
  data = np.zeros((num_samples, seq_length))
  for i in range(num_samples):
        (
            energies,
            exciton_energy, 
            exciton_linewidth, 
            fine_structure_splitting,
            power,
            polarizer_angle_deg, 
            noise,
            prominence
        ) = generate_parameter_variators(seq_length=seq_length, randomize=True)

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
        data[i] = spectrum / spectrum.max()

        parameters['energies'] = energies
        parameters['exciton_energy'] = exciton_energy
        parameters['exciton_linewidth'] = exciton_linewidth
        parameters['fine_structure_splitting'] = fine_structure_splitting
        parameters['power'] = power
        parameters['polarizer_angle_deg'] = polarizer_angle_deg
        parameters['noise'] = noise
        parameters['prominence'] = prominence

  return data, parameters

if __name__ == "__main__":
    num_samples = 100
    seq_length = 1000
    data = generate_data(num_samples, seq_length)
    qd_data = generate_qd_data(num_samples, seq_length)
    
    plt.figure(figsize=(12, 3))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(data[i], label='Random Peaks')
        plt.plot(qd_data[i], label='Quantum Dot')

    plt.legend()
    plt.title(f'Sample {i+1}')
    plt.show()