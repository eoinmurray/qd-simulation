import numpy as np

def generate_parameter_variators(
    seq_length=1000,
    randomize=False,
    energies_maxmin=(1580.0, 1600.0),
    exciton_energy_range=(1587.0, 1589.0),
    exciton_linewidth_range=(0.00001, 0.00004),
    fine_structure_splitting_range=(-0.03, 0.03),
    power_range=(2.0, 4.0),
    polarizer_angle_deg_range=(0.0, 90.0),
    noise_range=(0.0, 0.0),
    prominence_range=(0.01, 0.02),
    exciton_energy=1588.0,
    exciton_linewidth=0.01,
    fine_structure_splitting=0.0,
    power=3.0,
    polarizer_angle_deg=45.0,
    noise=0.0,
    prominence=0.1
):
    # Energy Range
    energies = np.linspace(energies_maxmin[0], energies_maxmin[1], seq_length)

    if randomize:
        exciton_energy = np.random.uniform(*exciton_energy_range)
        exciton_linewidth = np.random.uniform(*exciton_linewidth_range)
        fine_structure_splitting = np.random.uniform(*fine_structure_splitting_range)
        power = np.random.uniform(*power_range)
        polarizer_angle_deg = np.random.uniform(*polarizer_angle_deg_range)
        noise = np.random.uniform(*noise_range)
        prominence = np.random.uniform(*prominence_range)

    return (
        energies,
        exciton_energy, 
        exciton_linewidth, 
        fine_structure_splitting,
        power,
        polarizer_angle_deg, 
        noise,
        prominence
    )