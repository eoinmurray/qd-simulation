import streamlit as st
import numpy as np

import numpy as np
import streamlit as st

def generate_parameter_variators(seq_length=1000, randomize=False):
    # Energy Range
    energies_maxmin = st.sidebar.slider(
        "Energy Range [meV]",
        min_value=1570.0,
        max_value=1600.0,
        value=(1580.0, 1600.0),
        step=0.1
    )

    energies = np.linspace(energies_maxmin[0], energies_maxmin[1], seq_length)

    # Exciton Energy
    exciton_energy = st.sidebar.slider(
        "Exciton Energy (E_X) [meV]",
        min_value=1587.0,
        max_value=1589.0,
        value=np.random.uniform(1587.0, 1589.0) if randomize else 1588.0,
        step=0.1
    )

    # Exciton Linewidth
    exciton_linewidth = st.sidebar.slider(
        "Exciton Linewidth (gamma_X) [meV]",
        min_value=0.00001,
        max_value=0.4,
        value=np.random.uniform(0.00001, 0.0004) if randomize else 0.01,
        step=0.001
    )

    # Fine Structure Splitting
    fine_structure_splitting = st.sidebar.slider(
        "Fine Structure Splitting (FSS) [meV]",
        min_value=-0.3,
        max_value=0.3,
        value=np.random.uniform(-0.3, 0.3) if randomize else 0.0,
        step=0.001
    )

    # Power
    power = st.sidebar.slider(
        "Power",
        min_value=0.0001,
        max_value=4.0,
        value=np.random.uniform(2.5, 4.0) if randomize else 3.0,
        step=0.0001
    )

    # Polarizer Angle
    polarizer_angle_deg = st.sidebar.slider(
        "Polarizer Angle",
        min_value=0.0,
        max_value=90.0,
        value=np.random.uniform(0.0, 90.0) if randomize else 45.0,
        step=1.0
    )

    # Noise
    noise = st.sidebar.slider(
        "Noise",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        # value=np.random.uniform(0.0, 0.01) if randomize else 0.05,
        step=0.01
    )

    # Prominence
    prominence = st.sidebar.slider(
        "Prominence",
        min_value=0.01,
        max_value=10.0,
        value=np.random.uniform(0.01, 10.0) if randomize else 0.7,
        step=0.01,
        help="Required prominence of peaks."
    )

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