# Setup 🚀

## 1. Clone the repository

```bash
git clone https://github.com/cespeleta/novartis-datathon-2020
cd novartis-datathon-2020
```

## 2. Set up the Python environment

[Install conda](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) if not installed yet. Then create a conda environment using the `Makefile`. This command will create an environment called `novartis-datathon-2020`, as defined in the `environment.yml`

```bash
make create-environment
```

Next, activate the conda environment.

```bash
conda activate novartis-datathon-2020
```

Last, run export `PYTHONPATH=.` before executing any commands later on, or you will get errors like ModuleNotFoundError: No module named 'src'.

```
export PYTHONPATH=.
```
Setup done!

# How to use it

If we want to train the models we must follow the next steps:

1. Create features datasets

Datasets will be stored in: `novartis-datathon-2020/data/processed/`

```bash
make create-features
```

2. Train models

Models will stored in: `novartis-datathon-2020/models/`

```bash
make train-models
```

3. Execute the inference pipeline

Forecasts will be stored in `novartis-datathon-2020/model/s`

```bash
make forecast
```