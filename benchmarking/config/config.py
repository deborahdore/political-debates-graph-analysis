import os
from pathlib import Path

# ======================== utils ======================== #
valid_kge_models = ["TransE", "DistMult", "ComplEx", "HolE", "ConvE", "RotatE", "PairRE", "AutoSF", "BoxE", "TransH"]
nlp_models = ["bert"]
valid_noise_ratio = [0, 0.1, 0.2, 0.3, 1]

# ===================== Directories ===================== #
base_dir = os.path.abspath(".")
dataset_dir = os.path.join(base_dir, "dataset")
models_dir = os.path.join(base_dir, "models")
model_dir = os.path.join(models_dir, "{model}")
config_dir = os.path.join(base_dir, "config")
plot_dir = os.path.join(base_dir, "plot")
results_dir = os.path.join(base_dir, "evaluation_results")
noisy_dir = os.path.join(dataset_dir, "{noise}")
excel_dir = os.path.join(results_dir, "excel")

# ======================== Files ======================== #
original_dataset_file = os.path.join(dataset_dir, "original_dataset.csv")
triplets_file = os.path.join(dataset_dir, "triplets_file.tsv")
triplets_file_utils = os.path.join(dataset_dir, "{file_name}.json")
noisy_triples_file = os.path.join(noisy_dir, "triplets_file_{use}.tsv")
metrics_file = os.path.join(results_dir, "all/metrics_{model}_{ratio}.json")

# ================== Create Directories ================== #
assert Path(dataset_dir).exists()
Path(plot_dir).mkdir(parents=True, exist_ok=True)
Path(models_dir).mkdir(parents=True, exist_ok=True)
Path(config_dir).mkdir(parents=True, exist_ok=True)
Path(results_dir).mkdir(parents=True, exist_ok=True)
Path(excel_dir).mkdir(parents=True, exist_ok=True)
Path(os.path.join(results_dir, "all")).mkdir(parents=True, exist_ok=True)

for model in (valid_kge_models + nlp_models):
	for noise in valid_noise_ratio:
		model_dir_ratio = os.path.join(model_dir.format(model=model), str(noise))
		Path(model_dir_ratio).mkdir(exist_ok=True, parents=True)

for noise_ratio in valid_noise_ratio:
	Path(noisy_dir.format(noise=noise_ratio)).mkdir(exist_ok=True, parents=True)
