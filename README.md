# CICERO-SCM Surrogate

This repository hosts the surrogate modelling experiments that couple a multi-agent reinforcement learning setup with the CICERO simple climate model (SCM). We provide training scripts, data processing utilities, and exploratory notebooks that reproduce the surrogate analysis used in the accompanying paper submitted for AAMAS 2026.

## Repository Layout
- `src/`: source code for data generation, data processing, surrogate model, MARL environment wrappers, and training/analysis scripts.
- `ciceroscm/`: upstream CICERO SCM source, bundled here for convenience.
- `config/`: example experiment and training configurations.
- `notebooks/`: analysis of results and plotting utilities.
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
4. Install the remaining Python dependencies recorded in `requirements.txt`:
   `pip install -r requirements.txt`
   
## Usage
- Generate data using `python src/data_generation.py` which will load the `data_generation.yaml`. The data generation is based on CICERO-SCM and hence it will also load the `cicero_scm.yaml` file in which you can configure specific parameters for the CICERO-SCM.
   - Analyze the results of the data generation in `notebooks/input_data_analysis.ipynb` and update the path to your newly generated data.
- Process data for surrogate model training using `python src/data_processing.py` which will load the `data_processing.yaml`. Set the path to the directory of the data you want to process in the `data_processing.yaml` file.
- Train a surrogate model using `python src/train.py` which will load the `train.yaml`. Set the path to the directory of the processed data you want to train on in the `train.yaml` file.
   - Analyse the surrogate model in `notebooks/rnn_based_surrogates.ipynb` and update the path to your newly trained model.
- Run a marl experiment using `python src/marl_experiment.py` which will load the `marl.yaml`. Configure your MARL experiment in the `marl.yaml` file.
   - Analyse the MARL results in `notebooks/marl_results.ipynb` and update the path to the new results from your MARL experiment.
- Run a one-step inference speed test using `python src/speed_test.py` which will load the `speed_test.yaml`. Configure your speed test in the `speed_test.yaml` file.
- Run a policy consistency test using `python src/policy_consistency_test.py` which will load the `policy_consistency_test.yaml`. Configure which MARL run you want to test in the `policy_consistency_test.yaml` file.
- Run a policy preservation ordering test using `python src/policy_ordering_test.py` which will load the `policy_ordering_test.yaml`. Configure which MARL run you want to test in the `policy_ordering_test.yaml` file.

## Data & Results
Input datasets and generated results required for the paper are too large to keep in this repository. We can provide the full artefacts on request. Please reach out if you need access.

## Contributing
Feel free to open issues or pull requests for bug fixes and improvements. When adding experiments, include a brief description of the configuration and expected outputs so results remain reproducible.

