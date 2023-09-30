import os
from pathlib import Path

# ==================== Directories ==================== #
base_dir = os.path.abspath(".")
dataset_dir = os.path.join(base_dir, "dataset")
models_dir = os.path.join(base_dir, "models")
model_dir = os.path.join(models_dir, "{model}")
config_dir = os.path.join(base_dir, "config")
plot_dir = os.path.join(base_dir, "plot")
results_dir = os.path.join(base_dir, "results")

# ==================== Files ==================== #
original_dataset_file = os.path.join(dataset_dir, "original_dataset.csv")
triplets_file = os.path.join(dataset_dir, "triplets_file.tsv")

pipeline_config_file = os.path.join(config_dir, "pipeline_config.json")

# ==================== Models Names ==================== #
valid_kge_models = [
	"TransE",
	"DistMult",
	"TransH",
	"ComplEx",
	"HolE",
	"ConvE",
	"RotatE",
	"PairRE",
	"AutoSF",
	"BoxE",
]

# ==================== Create Directories ==================== #
assert Path(dataset_dir).exists()
Path(plot_dir).mkdir(parents=True, exist_ok=True)
Path(models_dir).mkdir(parents=True, exist_ok=True)
Path(config_dir).mkdir(parents=True, exist_ok=True)
Path(results_dir).mkdir(parents=True, exist_ok=True)

for model in valid_kge_models:
	Path(model_dir.format(model=model)).mkdir(exist_ok=True, parents=True)
