import argparse
from pathlib import Path

import rootutils
import torch.nn
from loguru import logger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from benchmarking import config
from benchmarking.evaluation import link_deletion, \
	link_prediction, \
	make_prediction, \
	relation_classification, \
	relation_prediction, \
	triple_classification
from benchmarking.training import optimization, training
from benchmarking.utils.dataset_utils import generate_mappings, \
	generate_noise, \
	generate_pretrained_embeddings, \
	generate_triplets


def get_kwargs():
	"""	Parse the command line arguments """

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

	parser.add_argument("--save_probabilities",
						default=False,
						action='store_true',
						help="whether or not to save probabilities for each label during prediction")

	parser.add_argument("--wandb_project_name",
						type=str,
						required=False,
						default=config.WANDB_PROJECT_NAME,
						help="wandb project name")

	parser.add_argument("--noise",
						type=int,
						required=False,
						default=0,
						help=f"Noise level to train the model on ({config.VALID_NOISE_RATIO})")

	parser.add_argument("--mode_node",
						type=str,
						required=False,
						default=[],
						help=f"Mode for creating connections between nodes: {config.VALID_MODE_NODE}")

	### REQUIRED ###
	parser.add_argument("--output_dir_name", type=str, required=True, help="Output directory name")

	parser.add_argument("--model", type=str, required=True, help=f"Model (name) to use ({config.VALID_MODELS})")

	parser.add_argument("--mode_text",
						type=str,
						required=True,
						help=f"Mode for creating feature nodes: {config.VALID_MODE_TEXT}")

	args = parser.parse_args()

	if args.mode_node:
		mode_nodes = [mode.strip() for mode in str(args.mode_node).split(",")]
		for mode in mode_nodes:
			assert mode in config.VALID_MODE_NODE
		config.MODE_NODE = mode_nodes

	assert args.mode_text in config.VALID_MODE_TEXT
	assert args.noise in config.VALID_NOISE_RATIO
	assert args.model in config.VALID_MODELS

	config.USE_PRETRAINED_EMBEDDINGS = args.use_pretrained_embeddings
	config.SPECIAL_BENCHMARKING_FLAG = args.special_benchmarking_flag
	config.WANDB = args.wandb
	config.WANDB_PROJECT_NAME = args.wandb_project_name
	config.MODE_TEXT = args.mode_text

	if args.generate:
		for noise in config.VALID_NOISE_RATIO:
			Path(config.noisy_dir.format(noise=noise)).mkdir(parents=True, exist_ok=True)

	Path(config.noisy_dir.format(noise=args.noise)).mkdir(parents=True, exist_ok=True)
	Path(config.plot_dir.format(task_name=args.output_dir_name, model=args.model, noise=args.noise)).mkdir(
		parents=True,
		exist_ok=True)

	logger.info(f"⚠️ Special benchmarking flag: {config.SPECIAL_BENCHMARKING_FLAG}")
	logger.info(f"⚠️ Use pretrained embeddings: {config.USE_PRETRAINED_EMBEDDINGS}")
	logger.info(f"⚠️ Mode Text: {config.MODE_TEXT}")
	logger.info(f"⚠️ Mode Node: {config.MODE_NODE}")
	logger.info(f"⚠️ Noise: {args.noise}")
	logger.info(f"⚠️ Model: {args.model}")

	return args


def evaluation(model: torch.nn.Module,
			   model_name: str,
			   results_dir: str,
			   noisy_split_triplets_file: str,
			   noisy_split_triplets_file2: str,
			   triplets_file_utils: str,
			   task_name: str,
			   ratio: int,
			   save_probabilities: bool = False):
	""" Evaluate a KGE model"""
	link_prediction(model=model,
					model_name=model_name,
					noisy_split_triplets_file=noisy_split_triplets_file,
					triplets_file_utils=triplets_file_utils,
					task_name=task_name,
					results_dir=results_dir,
					noise_ratio=ratio)
	link_deletion(model=model,
				  model_name=model_name,
				  noisy_split_triplets_file=noisy_split_triplets_file,
				  triplets_file_utils=triplets_file_utils,
				  task_name=task_name,
				  results_dir=results_dir,
				  noise_ratio=ratio)

	for relation_to_evaluate in [None, '__label__Support', '__label__Attack', '__label__Equivalent']:
		relation_prediction(model=model,
							model_name=model_name,
							noisy_split_triplets_file=noisy_split_triplets_file,
							triplets_file_utils=triplets_file_utils,
							task_name=task_name,
							results_dir=results_dir,
							noise_ratio=ratio,
							relation_to_evaluate=relation_to_evaluate)

	triple_classification(model=model,
						  model_name=model_name,
						  noisy_split_triplets_file=noisy_split_triplets_file,
						  triplets_file_utils=triplets_file_utils,
						  task_name=task_name,
						  results_dir=results_dir,
						  noise_ratio=ratio)

	threshold = relation_classification(model=model,
										model_name=model_name,
										noisy_split_triplets_file=noisy_split_triplets_file,
										triplets_file_utils=triplets_file_utils,
										task_name=task_name,
										results_dir=results_dir,
										noise_ratio=ratio)

	make_prediction(model=model,
					model_name=model_name,
					noisy_split_triplets_file2=noisy_split_triplets_file2,
					triplets_file_utils=triplets_file_utils,
					task_name=task_name,
					results_dir=results_dir,
					plot_dir=config.plot_dir,
					noise=ratio,
					threshold=threshold,
					save_probabilities=save_probabilities)


def main():
	args = get_kwargs()

	if args.generate:
		# generates triples from original file
		generate_triplets(original_dataset_file=config.original_dataset_file,
						  original_split_triplets_file=config.original_split_triplets_file,
						  noisy_split_triplets_file=config.noisy_split_triplets_file,
						  noisy_triplets_file=config.noisy_triplets_file)
		# generate node to label mappings
		generate_mappings(noisy_triplets_file=config.noisy_triplets_file,
						  triplets_file_utils=config.triplets_file_utils,
						  pretrained_embedding_file=config.pretrained_embedding_file)
		# generate embeddings
		generate_pretrained_embeddings(triplets_file_utils=config.triplets_file_utils,
									   pretrained_embedding_file=config.pretrained_embedding_file)

		# for every level of noise, create a dataset
		generate_noise(noisy_triplets_file=config.noisy_triplets_file,
					   noisy_split_triplets_file=config.noisy_split_triplets_file,
					   valid_noise=config.VALID_NOISE_RATIO)

	if args.optimize:
		optimization(model_name=args.model,
					 model_dir=config.model_dir,
					 noisy_split_triplets_file=config.noisy_split_triplets_file,
					 triplets_file_utils=config.triplets_file_utils,
					 task_name=args.output_dir_name,
					 noisy_triplets_file=config.noisy_triplets_file,
					 pretrained_embedding_file=config.pretrained_embedding_file)

	model = training(model_name=args.model,
					 model_dir=config.model_dir,
					 noisy_split_triplets_file=config.noisy_split_triplets_file,
					 noisy_triplets_file=config.noisy_triplets_file,
					 triplets_file_utils=config.triplets_file_utils,
					 task_name=args.output_dir_name,
					 ratio=args.noise,
					 pretrained_embedding_file=config.pretrained_embedding_file)

	evaluation(model=model,
			   model_name=args.model,
			   results_dir=config.results_dir,
			   noisy_split_triplets_file=config.noisy_split_triplets_file,
			   noisy_split_triplets_file2=config.noisy_split_triplets_file2,
			   triplets_file_utils=config.triplets_file_utils,
			   task_name=args.output_dir_name,
			   ratio=args.noise,
			   save_probabilities=args.save_probabilities)


if __name__ == '__main__':
	main()
