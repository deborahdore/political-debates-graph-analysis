import argparse
import os.path
import sys

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
	The get_kwargs function parses the command line arguments and returns them as a tuple.
	The function first checks that all the required arguments are present in sys.argv, then it
	parses each argument into its respective type (e.g., int, float, str). The function returns a
	tuple containing these parsed values.

	:return: A tuple of four elements:
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

	parser.add_argument("--model", type=str, required=True, help="Model (name) to use")

	parser.add_argument("--noise", type=int, required=True, help="Noise level to train the model on")

	args = parser.parse_args()

	assert args.noise in config.VALID_NOISE_RATIO
	assert args.model in config.VALID_MODELS

	return args.generate, args.optimize, args.noise, args.model


def bert_basic(model_name: str, noise: int):
	"""
	The bert_basic function is used to train a BERT model on the noisy triples file, and then evaluate it using link
	prediction, link deletion, and triple classification.

	:param model_name:str: Name the model
	:param noise:int: Specify the noise ratio
	"""
	model_file = os.path.join(config.model_dir.format(model="Bert"), f"{noise}/{model_name}_{noise}.pt")
	if not os.path.exists(model_file) or config.FORCE_TRAINING:
		model = bert_training(model_file=model_file,
							  model_name='Bert',
							  noisy_triples_file=config.noisy_triples_file,
							  ratio=noise)
	else:
		model = load_model(model_file)

	link_prediction_bert(model=model,
						 model_dir=config.model_dir.format(model="Bert"),
						 model_name='Bert',
						 noisy_triples_file=config.noisy_triples_file,
						 metrics_file=config.metrics_file.format(model=model_name, ratio=noise),
						 noise_ratio=noise)
	link_deletion_bert(model=model,
					   model_dir=config.model_dir.format(model="Bert"),
					   model_name='Bert',
					   noisy_triples_file=config.noisy_triples_file,
					   metrics_file=config.metrics_file.format(model=model_name, ratio=noise),
					   noise_ratio=noise)
	triple_classification_bert(model=model,
							   model_dir=config.model_dir.format(model="Bert"),
							   model_name='Bert',
							   noisy_triples_file=config.noisy_triples_file,
							   metrics_file=config.metrics_file.format(model=model_name, ratio=noise),
							   noise_ratio=noise)


def kge_basic(model_name: str, noise: int):
	"""
	The kge_basic function is a wrapper function that trains and evaluates the KGE model.

	:param model_name: str: Name the model
	:param noise: int: Specify the noise ratio of the dataset
	"""
	model_file = os.path.join(config.model_dir.format(model=model_name), f"{noise}/{model_name}_{noise}.pt")
	if not os.path.exists(model_file) or config.FORCE_TRAINING:
		result = training(model_dir=config.model_dir.format(model=model_name),
						  model_name=model_name,
						  model_file=model_file,
						  noisy_triples_file=config.noisy_triples_file,
						  triplets_file_utils=config.triplets_file_utils,
						  ratio=noise,
						  pretrained_embedding_file=config.pretrained_embedding_file)
		model = result.model

	else:
		model = load_model(model_file)

	link_prediction(model=model,
					noisy_triples_file=config.noisy_triples_file,
					triplets_file_utils=config.triplets_file_utils,
					model_name=model_name,
					metrics_file=config.metrics_file.format(model=model_name, ratio=noise),
					noise_ratio=noise)
	link_deletion(model=model,
				  model_name=model_name,
				  noisy_triples_file=config.noisy_triples_file,
				  triplets_file_utils=config.triplets_file_utils,
				  metrics_file=config.metrics_file.format(model=model_name, ratio=noise),
				  noise_ratio=noise)

	for relation_to_evaluate in [None, 'support', 'attack', 'equivalent']:
		relation_prediction(model=model,
							model_name=model_name,
							noisy_triples_file=config.noisy_triples_file,
							triplets_file_utils=config.triplets_file_utils,
							metrics_file=config.metrics_file.format(model=model_name, ratio=noise),
							noise_ratio=noise,
							relation_to_evaluate=relation_to_evaluate)


	triple_classification(model=model,
						  model_name=model_name,
						  noisy_triples_file=config.noisy_triples_file,
						  triplets_file_utils=config.triplets_file_utils,
						  metrics_file=config.metrics_file.format(model=model_name, ratio=noise),
						  noise_ratio=noise)

	relation_classification(model=model,
							model_name=model_name,
							noisy_triples_file=config.noisy_triples_file,
							triplets_file_utils=config.triplets_file_utils,
							metrics_file=config.metrics_file.format(model=model_name, ratio=noise),
							noise_ratio=noise)


if __name__ == '__main__':
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
		if model_name == 'Bert': exit(1)
		hyperparameter_optimization(model_name=model_name,
									model_dir=config.model_dir.format(model=model_name),
									noisy_triples_file=config.noisy_triples_file,
									triplets_file_utils=config.triplets_file_utils,
									pretrained_embedding_file=config.pretrained_embedding_file)

	else:
		if model_name == 'Bert':
			bert_basic(model_name, noise)
		else:
			kge_basic(model_name, noise)
