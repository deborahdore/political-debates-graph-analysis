import argparse
import os.path

import config
from evaluation import link_deletion, \
	link_deletion_bert, \
	link_prediction, \
	link_prediction_bert, \
	relation_classification, \
	relation_prediction, \
	triple_classification, \
	triple_classification_bert
from training import bert_training, hyperparameter_optimization, training
from utils.dataset_utils import generate_mappings, generate_noise, generate_triplets
from utils.utils import load_model


def get_kwargs():
	"""
	Parse the command line arguments

	:return:
	- generate argument containing True/False. If true, the datasets with noise will be generated
	- optimize argument containing True/False. If true, will perform hyperparameter optimization
	- model argument containing the model name to be optimized/trained
	- noise argument containing the noise ratio to optimize/train the model on
	"""

	parser = argparse.ArgumentParser()
	parser.add_argument("--generate",
						default=False,
						action='store_true',
						help="Whether or not to generate a new dataset")

	parser.add_argument("--optimize",
						default=False,
						action='store_true',
						help="Whether or not to perform hyper-parameter optimization on the model")

	parser.add_argument("--model", type=str, required=True, help=f"Model (name) to use ({config.VALID_MODELS})")

	parser.add_argument("--noise",
						type=int,
						required=True,
						help=f"Noise level to train the model on ({config.VALID_NOISE_RATIO})")

	args = parser.parse_args()

	assert args.noise in config.VALID_NOISE_RATIO
	assert args.model in config.VALID_MODELS

	return args.generate, args.optimize, args.noise, args.model


def bert_basic(ratio: int):
	"""
	Train and evaluate Bert model

	:param ratio:int: Specify the noise ratio of the dataset
	"""
	model_file = os.path.join(config.model_dir.format(model="bert"), f"{ratio}/bert_{ratio}.pt")

	if not os.path.exists(model_file) or config.FORCE_TRAINING:
		model = bert_training(model_file=model_file, noisy_triples_file=config.noisy_triples_file, ratio=ratio)
	else:
		model = load_model(model_file)

	link_prediction_bert(model=model,
						 noisy_triples_file=config.noisy_triples_file,
						 metrics_file=config.metrics_file.format(model="bert", ratio=ratio),
						 noise_ratio=ratio)

	link_deletion_bert(model=model,
					   noisy_triples_file=config.noisy_triples_file,
					   metrics_file=config.metrics_file.format(model="bert", ratio=ratio),
					   noise_ratio=ratio)

	triple_classification_bert(model=model,
							   model_dir=config.model_dir.format(model="bert"),
							   model_name='bert',
							   noisy_triples_file=config.noisy_triples_file,
							   metrics_file=config.metrics_file.format(model="bert", ratio=ratio),
							   noise_ratio=ratio)


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
	generate_dataset, optimization, noise, model_name = get_kwargs()

	if generate_dataset:
		# generates triples from original file
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
	if optimization:
		if model_name == 'bert': exit(1)
		hyperparameter_optimization(model_name=model_name,
									model_dir=config.model_dir.format(model=model_name),
									noisy_triples_file=config.noisy_triples_file,
									triplets_file_utils=config.triplets_file_utils,
									pretrained_embedding_file=config.pretrained_embedding_file)

	else:
		if model_name == 'bert':
			bert_basic(noise)
		else:
			kge_basic(model_name, noise)


if __name__ == '__main__':
	main()
