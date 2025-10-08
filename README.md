# CICERO-SCM Surrogate

This repository hosts the surrogate modelling experiments that couple a multi-agent reinforcement learning setup with the CICERO simple climate model (SCM). We provide training scripts, data processing utilities, and exploratory notebooks that reproduce the surrogate analysis used in the accompanying paper.

## Repository Layout
- `src/`: surrogate model, MARL environment wrappers, and training/analysis scripts.
- `ciceroscm/`: upstream CICERO SCM source, bundled for convenience.
- `config/`: example experiment and training configurations.
- `notebooks/`: exploratory analysis and plotting utilities.
- `data/`: placeholders for input data and generated artefacts (not versioned in full because of size).

## Installation
1. Create the base environment (Python 3.10):  
   `conda env create -f environment.yaml`  
   `conda activate cicero`
2. Install the CICERO SCM dependency in editable mode so the surrogate code can import it:  
   `cd ciceroscm`  
   `pip install -e .`
3. Return to the repository root for running scripts:  
   `cd ..`

(Optional) If you plan to edit the surrogate sources directly, add the repository to your `PYTHONPATH` or install it in editable mode once packaging metadata is added.

## Usage
- Launch exploratory notebooks: `jupyter lab notebooks`
- Train a surrogate model with a configuration file, for example:  
  `python src/train.py --config config/example.yaml`
- Utilities for data preprocessing and evaluation live under `src/utils/`.

## Data & Results
Input datasets and generated results required for the paper are too large to keep in this repository. We can provide the full artefacts on request; please reach out if you need access.

## Contributing
Feel free to open issues or pull requests for bug fixes and improvements. When adding experiments, include a brief description of the configuration and expected outputs so results remain reproducible.

