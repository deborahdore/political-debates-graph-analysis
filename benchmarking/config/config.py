import os
from pathlib import Path

# ======================== utils ======================== #
valid_kge_models = ["TransE", "DistMult", "ComplEx", "HolE", "ConvE", "RotatE", "PairRE", "AutoSF", "BoxE", "TransH"]
noise = [0.5, 0.1, 0.15, 0.2, 0.3, 1]
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

# ======================== Files ======================== #
original_dataset_file = os.path.join(dataset_dir, "original_dataset.csv")
triplets_file = os.path.join(dataset_dir, "triplets_file_{use}.tsv")

# ================== Create Directories ================== #
assert Path(dataset_dir).exists()
Path(plot_dir).mkdir(parents=True, exist_ok=True)
Path(models_dir).mkdir(parents=True, exist_ok=True)
Path(config_dir).mkdir(parents=True, exist_ok=True)

for model in valid_kge_models:
	Path(model_dir.format(model=model)).mkdir(exist_ok=True, parents=True)
