# Replicate QD spectrum

## Installation

1. First install [miniconda](https://docs.anaconda.com/miniconda/).
2. In a terminal now equipped with the `conda` command run:

```
conda create spectrum
conda activate spectrum
conda install streamlit matplotlib numpy scipy watchdog
streamlit run simulation/streamlit_app.py
```