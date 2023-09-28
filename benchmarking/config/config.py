import os
from pathlib import Path

# ==================== Directories ==================== #
base_dir = os.path.abspath(".")
dataset_dir = os.path.join(base_dir, "dataset")
output_dir = os.path.join(base_dir, "output")
models_dir = os.path.join(base_dir, "models")
model_dir = os.path.join(models_dir, "{model}")
noisy_dataset_dir = os.path.join(dataset_dir, "noisy_dataset_{ratio}")
config_dir = os.path.join(base_dir, "config")
plot_dir = os.path.join(base_dir, "plot")
checkpoint_dir = os.path.join(output_dir, "checkpoint")

# ==================== Files ==================== #
original_dataset_file = os.path.join(dataset_dir, "relation_graph.csv")
dataset_file = os.path.join(dataset_dir, "dataset.csv")
noisy_dataset_file = noisy_dataset_dir + "/" + "noisy_{ratio}_{use}.tsv"
pipeline_config_file = os.path.join(config_dir, "pipeline_config.json")
plot_file = os.path.join(plot_dir, "{name}-{ratio}.svg")
model_file = os.path.join(model_dir, "{model}-{ratio}.pt")
metric_file = model_dir + "/metrics-{ratio}.json"

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
# ==================== Ratio of noise ==================== #
ratio = [0.05, 0.1, 0.2, 0.3, 1]

# ==================== Create Directories ==================== #
assert Path(dataset_dir).exists()

Path(output_dir).mkdir(parents = True, exist_ok = True)
Path(checkpoint_dir).mkdir(parents = True, exist_ok = True)
Path(plot_dir).mkdir(parents = True, exist_ok = True)

for model in valid_kge_models:
	Path(model_dir.format(model = model)).mkdir(exist_ok = True, parents = True)

for r in ratio:
	Path(noisy_dataset_dir.format(ratio = r)).mkdir(exist_ok = True, parents = True)
