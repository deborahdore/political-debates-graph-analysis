import os

# ----------- directories ----------- #
base_dir = os.path.abspath(".")
dataset_dir = os.path.join(base_dir, "dataset")

# ----------- files ----------- #
original_dataset_file = os.path.join(dataset_dir, "relation_graph.csv")
dataset_file = os.path.join(dataset_dir, "dataset.csv")
noisy_dataset_file = os.path.join(dataset_dir, "noisy_dataset_{ratio}.csv")
