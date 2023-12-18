import os
from pathlib import Path

import torch

# ======================== utils ======================== #
VALID_MODELS = ["TransE", "DistMult", "ComplEx", "HolE", "ConvE", "PairRE", "AutoSF", "RotatE", "TransH", "BoxE",
				"Bert"]
VALID_NOISE_RATIO = [0, 10, 20, 30, 100]

# ===================== Settings ===================== #

SPECIAL_BENCHMARKING_FLAG = True
USE_PRETRAINED_EMBEDDINGS = True
FORCE_TRAINING = True
NUM_TRIALS = 30

WANDB_PROJECT_NAME = "kge_trial_1"

MODE_TEXT = 'text'
MODE_NODE = ['speaker', 'year']

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ===================== Directories ===================== #

base_dir = os.path.abspath(".")
dataset_dir = os.path.join(base_dir, "dataset")
models_dir = os.path.join(base_dir, "models")
model_dir = os.path.join(models_dir, "{model}")
results_dir = os.path.join(base_dir, "evaluation_results")
noisy_dir = os.path.join(dataset_dir, "{noise}")
excel_dir = os.path.join(results_dir, "excel")

# ======================== Files ======================== #

original_dataset_file = os.path.join(dataset_dir, "original_dataset.csv")
original_triplets_file = os.path.join(dataset_dir, "original_triplets_file.tsv")
triplets_file = os.path.join(dataset_dir, "triplets_file.tsv")
triplets_file_utils = os.path.join(dataset_dir, "{file_name}.json")
pretrained_embedding_file = os.path.join(dataset_dir, "pretrained_embedding_file.npy")
noisy_triples_file = os.path.join(noisy_dir, "triplets_file_{use}.tsv")
original_noisy_triples_file = os.path.join(noisy_dir, "original_triplets_file_{use}.tsv")
metrics_file = os.path.join(results_dir, "all/metrics_{model}_{ratio}.json")

# ================== Asserts ================== #

assert Path(dataset_dir).exists()
assert MODE_TEXT is not None

# ================== Create Directories ================== #

Path(models_dir).mkdir(parents=True, exist_ok=True)
Path(results_dir).mkdir(parents=True, exist_ok=True)
Path(excel_dir).mkdir(parents=True, exist_ok=True)
Path(os.path.join(results_dir, "all")).mkdir(parents=True, exist_ok=True)

for model in VALID_MODELS:
	for noise in VALID_NOISE_RATIO:
		model_dir_ratio = os.path.join(model_dir.format(model=model), str(noise))
		Path(model_dir_ratio).mkdir(exist_ok=True, parents=True)

for noise_ratio in VALID_NOISE_RATIO:
	Path(noisy_dir.format(noise=noise_ratio)).mkdir(exist_ok=True, parents=True)
