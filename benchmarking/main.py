import ast
import os.path
import sys

import torch
from loguru import logger

from config import config
from config.config import (metrics_file,
						   model_dir,
						   noisy_triples_file,
						   original_dataset_file,
						   triplets_file,
						   triplets_file_utils,
						   valid_models,
						   valid_noise_ratio, )
from evaluation import link_deletion, \
	link_deletion_bert, \
	link_prediction, \
	link_prediction_bert, \
	triple_classification, \
	triple_classification_bert
from training import bert_training, hyperparameter_optimization, training
from utils.dataset_utils import generate_mappings, generate_noise, generate_triplets
from utils.utils import load_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def parse_command_line():
	"""
	The parse_command_line function parses the command line arguments and returns them as a tuple.
	The function first checks that all of the required arguments are present in sys.argv, then it
	parses each argument into its respective type (e.g., int, float, str). The function returns a
	tuple containing these parsed values.

	:return: A tuple of four elements:
	- generate argument containing True/False. If true, the datasets with noise will be generated
	- optimize argument containing True/False. If true, will perform hyperparameter optimization
	- model argument containing the model name to be optimized/trained
	- noise argument containing the noise ratio to optimize/train the model on

	"""
	if len(sys.argv) < 8:
		logger.error("missing arguments from command line")
		raise Exception("missing arguments from command line")

	assert "--generate" in sys.argv
	assert "--optimize" in sys.argv
	assert "--model" in sys.argv
	assert "--noise" in sys.argv

	gen_arg = ast.literal_eval(sys.argv[sys.argv.index("--generate") + 1])
	opt_arg = ast.literal_eval(sys.argv[sys.argv.index("--optimize") + 1])
	noise_arg = int(sys.argv[sys.argv.index("--noise") + 1])
	model_arg = sys.argv[sys.argv.index("--model") + 1]
	return gen_arg, opt_arg, noise_arg, model_arg


def bert_basic(model_name: str, noise: int, force_retrain: bool = True):
	"""
	The bert_basic function is used to train a BERT model on the noisy triples file, and then evaluate it using link
	prediction, link deletion, and triple classification.

	:param model_name:str: Name the model
	:param noise:int: Specify the noise ratio
	"""
	# check if model exists
	model_file = os.path.join(model_dir.format(model="Bert"), f"{noise}/{model_name}_{noise}.pt")
	if not os.path.exists(model_file) or force_retrain:
		# bert training
		model = bert_training(model_file=model_file,
							  model_name='Bert',
							  noisy_triples_file=noisy_triples_file,
							  ratio=noise)
	else:
		model = load_model(model_file, device)

	# evaluation
	link_prediction_bert(model=model,
						 model_dir=model_dir.format(model="Bert"),
						 model_name='Bert',
						 noisy_triples_file=noisy_triples_file,
						 metrics_file=metrics_file.format(model=model_name, ratio=noise),
						 noise_ratio=noise)
	link_deletion_bert(model=model,
					   model_dir=model_dir.format(model="Bert"),
					   model_name='Bert',
					   noisy_triples_file=noisy_triples_file,
					   metrics_file=metrics_file.format(model=model_name, ratio=noise),
					   noise_ratio=noise)
	triple_classification_bert(model=model,
							   model_dir=model_dir.format(model="Bert"),
							   model_name='Bert',
							   noisy_triples_file=noisy_triples_file,
							   metrics_file=metrics_file.format(model=model_name, ratio=noise),
							   noise_ratio=noise)


def kge_basic(model_name: str, noise: int, force_retrain: bool = True):
	"""
	The kge_basic function is a wrapper function that trains and evaluates the KGE model.

	:param force_retrain: bool : Force model's training
	:param model_name: str: Name the model
	:param noise: int: Specify the noise ratio of the dataset
	"""
	# NOTE: ALWAYS RETRAIN THE MODEL WHEN EVALUATING TO BE SURE THE PREVIOUS MODEL WAS NOT TRAINED ON ANOTHER DATASET
	model_file = os.path.join(model_dir.format(model=model_name), f"{noise}/{model_name}_{noise}.pt")
	if not os.path.exists(model_file) or force_retrain:
		result = training(model_dir=model_dir.format(model=model_name),
						  model_name=model_name,
						  model_file=model_file,
						  noisy_triples_file=noisy_triples_file,
						  triplets_file_utils=triplets_file_utils,
						  ratio=noise)
		model = result.model

	else:
		model = load_model(model_file, device)

	# evaluation
	link_prediction(model=model,
					noisy_triples_file=noisy_triples_file,
					triplets_file_utils=triplets_file_utils,
					model_name=model_name,
					metrics_file=metrics_file.format(model=model_name, ratio=noise),
					noise_ratio=noise)
	link_deletion(model=model,
				  model_name=model_name,
				  noisy_triples_file=noisy_triples_file,
				  triplets_file_utils=triplets_file_utils,
				  metrics_file=metrics_file.format(model=model_name, ratio=noise),
				  noise_ratio=noise)
	triple_classification(model=model,
						  model_name=model_name,
						  noisy_triples_file=noisy_triples_file,
						  triplets_file_utils=triplets_file_utils,
						  metrics_file=metrics_file.format(model=model_name, ratio=noise),
						  noise_ratio=noise)


if __name__ == '__main__':
	# Command line must contain 4 arguments: generate, optimize, model_name, noise
	generate_dataset, optimization, noise, model_name = parse_command_line()

	assert noise in valid_noise_ratio
	assert model_name in valid_models

	if generate_dataset:
		# generates triples from original file
		generate_triplets(original_dataset_file=original_dataset_file, triples_file=triplets_file)
		# generate node to label mappings
		generate_mappings(triplets_file=triplets_file, triplets_file_utils=triplets_file_utils)
		# for every level of noise, create a dataset
		generate_noise(triplets_file=triplets_file,
					   noisy_triples_file=noisy_triples_file,
					   valid_noise=config.valid_noise_ratio)
	if optimization:
		if model_name == 'Bert': exit(1)
		hyperparameter_optimization(model_name=model_name,
									model_dir=model_dir.format(model=model_name),
									noisy_triples_file=noisy_triples_file,
									triplets_file_utils=triplets_file_utils)

	else:
		if model_name == 'Bert':
			bert_basic(model_name, noise)
		else:
			kge_basic(model_name, noise)

# process_results()
