import argparse
import os.path

import config
from evaluation import link_deletion, \
	link_prediction, \
	relation_classification, \
	relation_prediction, \
	triple_classification
from training import hyperparameter_optimization, training
from utils.dataset_utils import generate_mappings, generate_noise, generate_triplets
from utils.utils import load_model


def get_kwargs():
	"""
	Parse the command line arguments

	:return:
	- Generate argument containing True/False. If true, the datasets with noise will be generated
	- optimize argument containing True/False. If true, will perform hyperparameter optimization
	- model argument containing the model name to be optimized/trained
	- noise argument containing the noise ratio to optimize/train the model on
	"""

	parser = argparse.ArgumentParser()
	### NOT REQUIRED ###
	parser.add_argument("--generate",
						default=False,
						action='store_true',
						help="whether or not to generate a new dataset")

	parser.add_argument("--optimize",
						default=False,
						action='store_true',
						help="whether or not to perform hyper-parameter optimization on the model")

	parser.add_argument("--special_benchmarking_flag",
						default=False,
						action='store_true',
						help="whether or not to perform evaluation with only support, attack and equivalent relations")

	parser.add_argument("--use_pretrained_embeddings",
						default=False,
						action='store_true',
						help="whether or not to use pretrained embeddings for the KGE models")

	parser.add_argument("--wandb", default=False, action='store_true', help="whether or not to use wandb for logging")

	parser.add_argument("--wandb_project_name",
						type=str,
						required=False,
						default=config.WANDB_PROJECT_NAME,
						help="wandb project name")

	### REQUIRED ###
	parser.add_argument("--model", type=str, required=True, help=f"Model (name) to use ({config.VALID_MODELS})")

	parser.add_argument("--noise",
						type=int,
						required=True,
						help=f"Noise level to train the model on ({config.VALID_NOISE_RATIO})")

	parser.add_argument("--mode_text",
						type=str,
						required=True,
						help=f"Mode for creating feature nodes: {config.VALID_MODE_TEXT}")

	parser.add_argument("--mode_node",
						type=str,
						required=True,
						help=f"Mode for creating connections between nodes: {config.VALID_MODE_NODE}")

	args = parser.parse_args()

	config.USE_PRETRAINED_EMBEDDINGS = args.use_pretrained_embeddings
	config.WANDB = args.wandb
	config.WANDB_PROJECT_NAME = args.wandb_project_name

	config.MODE_TEXT = args.mode_text
	config.MODE_NODE = [mode.strip() for mode in str(args.mode_node).split(",")]

	assert config.MODE_TEXT in config.VALID_MODE_TEXT
	for mode in config.MODE_NODE:
		assert mode in config.VALID_MODE_NODE

	assert args.noise in config.VALID_NOISE_RATIO
	assert args.model in config.VALID_MODELS

	return args


def kge_basic(name: str, ratio: int):
	"""
	Trains and evaluate a KGE model

	:param name: str: Name the model to train/evaluate
	:param ratio: int: Specify the noise ratio of the dataset
	"""
	model_file = os.path.join(config.model_dir.format(model=name), f"{ratio}/{name}_{ratio}.pt")
	if not os.path.exists(model_file) or config.FORCE_TRAINING:
		result = training(model_dir=config.model_dir.format(model=name),
						  model_name=name,
						  model_file=model_file,
						  noisy_triples_file=config.noisy_triples_file,
						  triplets_file_utils=config.triplets_file_utils,
						  ratio=ratio,
						  pretrained_embedding_file=config.pretrained_embedding_file)
		model = result.model

	else:
		model = load_model(model_file)

	link_prediction(model=model,
					noisy_triples_file=config.noisy_triples_file,
					triplets_file_utils=config.triplets_file_utils,
					model_name=name,
					metrics_file=config.metrics_file.format(model=name, ratio=ratio),
					noise_ratio=ratio)
	link_deletion(model=model,
				  model_name=name,
				  noisy_triples_file=config.noisy_triples_file,
				  triplets_file_utils=config.triplets_file_utils,
				  metrics_file=config.metrics_file.format(model=name, ratio=ratio),
				  noise_ratio=ratio)

	for relation_to_evaluate in [None, 'support', 'attack', 'equivalent']:
		relation_prediction(model=model,
							model_name=name,
							noisy_triples_file=config.noisy_triples_file,
							triplets_file_utils=config.triplets_file_utils,
							metrics_file=config.metrics_file.format(model=name, ratio=ratio),
							noise_ratio=ratio,
							relation_to_evaluate=relation_to_evaluate)

	triple_classification(model=model,
						  model_name=name,
						  noisy_triples_file=config.noisy_triples_file,
						  triplets_file_utils=config.triplets_file_utils,
						  metrics_file=config.metrics_file.format(model=name, ratio=ratio),
						  noise_ratio=ratio)

	relation_classification(model=model,
							model_name=name,
							noisy_triples_file=config.noisy_triples_file,
							triplets_file_utils=config.triplets_file_utils,
							metrics_file=config.metrics_file.format(model=name, ratio=ratio),
							noise_ratio=ratio)


def main():
	""" Main function """
	args = get_kwargs()

	if args.generate:
		# generates triples from original file
		if os.path.exists(config.original_dataset_file):
			generate_triplets(original_dataset_file=config.original_dataset_file,
							  triples_file=config.triplets_file,
							  original_triplets_file=config.original_triplets_file)
		# generate node to label mappings
		generate_mappings(triplets_file=config.triplets_file,
						  triplets_file_utils=config.triplets_file_utils,
						  pretrained_embedding_file=config.pretrained_embedding_file)
		# for every level of noise, create a dataset
		generate_noise(triplets_file=config.triplets_file,
					   original_triplets_file=config.original_triplets_file,
					   noisy_triples_file=config.noisy_triples_file,
					   valid_noise=config.VALID_NOISE_RATIO)
	if args.optimize:
		if args.model == 'bert': exit(1)
		hyperparameter_optimization(model_name=args.model,
									model_dir=config.model_dir.format(model=args.model),
									noisy_triples_file=config.noisy_triples_file,
									triplets_file_utils=config.triplets_file_utils,
									pretrained_embedding_file=config.pretrained_embedding_file)

	else:
		kge_basic(args.model, args.noise)


if __name__ == '__main__':
	main()
