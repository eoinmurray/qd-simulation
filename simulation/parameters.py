import math
import numpy as np

def generate_parameters(
    power,
    polarizer_angle_deg,
    energies,
    exciton_energy=1588.0, 
    exciton_linewidth=0.01, 
    fine_structure_splitting=0.0, 
):
    particles = [
        {
            "name": "X",
            "energy": exciton_energy,
            "energy_formula": "e_x",
            "linewidth": exciton_linewidth,
            "fss": fine_structure_splitting,
            "power_dependence": 1.3,
            "intensity_scaling": 1.0,
        },
        {
            "name": "X+",
            "energy": exciton_energy - 9.84153 + (0.00485 * exciton_energy),
            "energy_formula": "e_x - 9.84153 + (0.00485 * e_x)",
            "linewidth": exciton_linewidth,
            "fss": fine_structure_splitting,
            "power_dependence": 2.7,
            "intensity_scaling": 1.0,
        },
        {
            "name": "X+hot",
            "energy": exciton_energy - 8.40923 + (0.0037 * exciton_energy),
            "energy_formula": "e_x - 8.40923 + (0.0037 * e_x)",
            "linewidth": exciton_linewidth,
            "fss": fine_structure_splitting,
            "power_dependence": 1.5,
            "intensity_scaling": 1.0,
        },
        {
            "name": "X+hot+excited",
            "energy": exciton_energy + 3.0,
            "energy_formula": "e_x + 3.0",
            "linewidth": exciton_linewidth,
            "fss": fine_structure_splitting,
            "power_dependence": 1.0,
            "intensity_scaling": 1.0,
        },
        {
            "name": "X* (excited neutral)",
            "energy": exciton_energy - 138.2527 + (0.09067 * exciton_energy),
            "energy_formula": "e_x - 138.2527 + (0.09067 * e_x)",
            "linewidth": exciton_linewidth,
            "fss": fine_structure_splitting,
            "power_dependence": 1.0,
            "intensity_scaling": 1.0,
        },
        {
            "name": "XX",
            "energy": exciton_energy + 11.77738 - (0.00991 * exciton_energy),
            "energy_formula": "e_x + 11.77738 - (0.00991 * e_x)",
            "linewidth": exciton_linewidth,
            "fss": fine_structure_splitting,
            "power_dependence": 2.1,
            "intensity_scaling": 1.0,
        },
        {
            "name": "XX+",
            "energy": exciton_energy - 4.8,
            "energy_formula": "e_x - 4.8",
            "linewidth": exciton_linewidth,
            "fss": fine_structure_splitting,
            "power_dependence": 2.0,
            "intensity_scaling": 1.0,
        },
    ]

    parameters = {
        "particles": particles,
        "power": power,
        "polarizer_angle_deg": polarizer_angle_deg,
        "energies": energies,
    }
    
    return parameters