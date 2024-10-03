import streamlit as st
import numpy as np

def generate_parameter_variators():
  # energies = np.linspace(1580.0, 1590.0, 1000)
  energies_maxmin = st.sidebar.slider(
      "Energy Range [meV]",
      min_value=1570.0,
      max_value=1600.0,
      value=(1580.0, 1600.0),
      step=0.1
  )

  energies = np.linspace(energies_maxmin[0], energies_maxmin[1], 1000)

  exciton_energy = st.sidebar.slider(
      "Exciton Energy (E_X) [meV]",
      min_value=1587.0,
      max_value=1589.0,
      value=1588.0,
      step=0.1
  )

  exciton_linewidth = st.sidebar.slider(
      "Exciton Linewidth (gamma_X) [meV]",
      min_value=0.00001,
      max_value=0.4,
      value=0.01,
      step=0.001
  )

  fine_structure_splitting = st.sidebar.slider(
      "Fine Structure Splitting (FSS) [meV]",
      min_value=-0.3,
      max_value=0.3,
      value=0.0,
      step=0.001
  )

  power = st.sidebar.slider(
      "Power",
      min_value=0.0001,
      max_value=4.0,
      value=3.0,
      step=0.0001
  )

  polarizer_angle_deg = st.sidebar.slider(
      "Polarizer Angle",
      min_value=0.0,
      max_value=90.0,
      value=45.0,
      step=1.0
  )

  noise = st.sidebar.slider(
      "Noise",
      min_value=0.0,
      max_value=1.0,
      value=0.05,
      step=0.01
  )

  prominence = st.sidebar.slider(
    "Prominence",
    min_value=0.01,
    max_value=10.0,
    value=0.7,
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