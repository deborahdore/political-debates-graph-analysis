import os
from pathlib import Path

import torch

# ======================== utils ======================== #
# semantic matching: DistMulT , HolE, RESCAL
# transactional: TransE, TransH, TransD
# other: ConvE, ConvKB

VALID_MODELS = ["TransE", "TransH", "TransD", "DistMult", "RESCAL", "HolE", "ConvE", "ConvKB"]
VALID_NOISE_RATIO = [0, 10, 20, 30, 100]

# ===================== Settings ===================== #

SPECIAL_BENCHMARKING_FLAG = False
USE_PRETRAINED_EMBEDDINGS = False
FORCE_TRAINING = True
NUM_TRIALS = 30
SEED = 42

SPLIT = ['train', 'dev', 'test']

WANDB = False
WANDB_PROJECT_NAME = "KGE Experiments"

MODE_TEXT = 'text'
MODE_NODE = ['claim+premise', 'year']

VALID_MODE_TEXT = ['text', 'text+type']
VALID_MODE_NODE = ['claim+premise', 'speaker', 'year']

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ===================== Directories ===================== #

base_dir = os.path.abspath(".")
output_dir = os.path.join(base_dir, "output/{task_name}")
model_dir = os.path.join(output_dir, "{model}")
results_dir = os.path.join(model_dir, "results")
plot_dir = os.path.join(results_dir, "plot")

dataset_dir = os.path.join(base_dir, "dataset")
dataset2_dir = os.path.join(base_dir, "dataset2")
noisy_dir = os.path.join(dataset_dir, "noise_{noise}")

# ======================== Files ======================== #

original_dataset_file = os.path.join(dataset_dir, "original_dataset.csv")
original_split_triplets_file = os.path.join(dataset_dir, "original_{split}_relations.tsv")
original_triplets_file = os.path.join(dataset_dir, "original_relations.tsv")

triplets_file_utils = os.path.join(dataset_dir, "{file_name}.json")
pretrained_embedding_file = os.path.join(dataset_dir, "pretrained_embedding_file.npy")

noisy_split_triplets_file = os.path.join(noisy_dir, "{split}_relations.tsv")
noisy_split_triplets_file2 = os.path.join(dataset2_dir, "{split}_relations.tsv")

noisy_triplets_file = os.path.join(noisy_dir, "relations.tsv")

# ================== Asserts ================== #

assert Path(dataset_dir).exists()
assert MODE_TEXT is not None
