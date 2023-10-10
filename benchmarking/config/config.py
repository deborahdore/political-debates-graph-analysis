import os
from pathlib import Path

# ======================== utils ======================== #
valid_kge_models = ["TransE", "DistMult", "ComplEx", "HolE", "ConvE", "RotatE", "PairRE", "AutoSF", "BoxE", "TransH"]
valid_noise_ratio = [0, 0.1, 0.2, 0.3, 1]
link_metrics = ['mr', 'mrr', 'hits_at_1', 'hits_at_3', 'hits_at_5', 'hits_at_10']
triple_metrics = ['f1_macro', 'f1_neg', 'f1_pos', 'norm_dist', 'z_stat']
strategies1 = ['head', 'tail', 'both']
strategies2 = ['optimistic', 'pessimistic', 'realistic']

# ===================== Directories ===================== #
base_dir = os.path.abspath(".")
dataset_dir = os.path.join(base_dir, "dataset")
models_dir = os.path.join(base_dir, "models")
model_dir = os.path.join(models_dir, "{model}")
config_dir = os.path.join(base_dir, "config")
plot_dir = os.path.join(base_dir, "plot")
results_dir = os.path.join(base_dir, "evaluation_results")
noisy_dir = os.path.join(dataset_dir, "{noise}")
# ======================== Files ======================== #
original_dataset_file = os.path.join(dataset_dir, "original_dataset.csv")
triplets_file = os.path.join(dataset_dir, "triplets_file.tsv")
noisy_triples_file = os.path.join(noisy_dir, "triplets_file_{use}.tsv")
metrics_file = os.path.join(results_dir, "metrics_{model}_{ratio}.json")

# ================== Create Directories ================== #
assert Path(dataset_dir).exists()
Path(plot_dir).mkdir(parents=True, exist_ok=True)
Path(models_dir).mkdir(parents=True, exist_ok=True)
Path(config_dir).mkdir(parents=True, exist_ok=True)
Path(results_dir).mkdir(parents=True, exist_ok=True)

for model in valid_kge_models:
	Path(model_dir.format(model=model)).mkdir(exist_ok=True, parents=True)

for noise_ratio in valid_noise_ratio:
	Path(noisy_dir.format(noise=noise_ratio)).mkdir(exist_ok=True, parents=True)
