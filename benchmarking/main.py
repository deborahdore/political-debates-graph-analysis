import ast
import sys

import torch
from loguru import logger

from config import config
from config.config import (metrics_file,
						   model_dir,
						   noisy_triples_file,
						   original_dataset_file,
						   plot_dir,
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
from utils.results_utils import process_results

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


def main():
	# Command line must contain 4 arguments: generate, optimize, model, noise
	generate_dataset, optimization, noise, model = parse_command_line()

	if generate_dataset:
		# generates triples from original file
		generate_triplets(original_dataset_file=original_dataset_file, triples_file=triplets_file)
		# generate node to label mappings
		generate_mappings(triplets_file=triplets_file, triplets_file_utils=triplets_file_utils)
		# for every level of noise, create a dataset
		generate_noise(triplets_file=triplets_file,
					   noisy_triples_file=noisy_triples_file,
					   valid_noise=config.valid_noise_ratio)

	if not optimization:
		logger.info("# -------- basic training with best pipeline config -------- # \n ")
		assert noise in valid_noise_ratio
		assert model in valid_models

		try:
			if model != "Bert":
				result = training(model_dir=model_dir.format(model=model),
								  model_name=model,
								  noisy_triples_file=noisy_triples_file,
								  triplets_file_utils=triplets_file_utils,
								  plot_dir=plot_dir,
								  ratio=noise)

				link_prediction(result=result,
								noisy_triples_file=noisy_triples_file,
								triplets_file_utils=triplets_file_utils,
								model_name=model,
								metrics_file=metrics_file.format(model=model, ratio=noise),
								noise_ratio=noise)

				link_deletion(result=result,
							  model_name=model,
							  noisy_triples_file=noisy_triples_file,
							  triplets_file_utils=triplets_file_utils,
							  metrics_file=metrics_file.format(model=model, ratio=noise),
							  noise_ratio=noise)

				triple_classification(result=result,
									  model_name=model,
									  noisy_triples_file=noisy_triples_file,
									  triplets_file_utils=triplets_file_utils,
									  metrics_file=metrics_file.format(model=model, ratio=noise),
									  noise_ratio=noise)
			else:

				model = bert_training(model_dir=model_dir.format(model="bert"),
									  model_name='bert',
									  noisy_triples_file=noisy_triples_file,
									  ratio=noise)

				link_prediction_bert(model=model,
									 model_dir=model_dir.format(model="bert"),
									 model_name='bert',
									 noisy_triples_file=noisy_triples_file,
									 metrics_file=metrics_file.format(model=model, ratio=noise),
									 noise_ratio=noise)

				link_deletion_bert(model=model,
								   model_dir=model_dir.format(model="bert"),
								   model_name='bert',
								   noisy_triples_file=noisy_triples_file,
								   metrics_file=metrics_file.format(model=model, ratio=noise),
								   noise_ratio=noise)

				triple_classification_bert(model=model,
										   model_dir=model_dir.format(model="bert"),
										   model_name='bert',
										   noisy_triples_file=noisy_triples_file,
										   metrics_file=metrics_file.format(model=model, ratio=noise),
										   noise_ratio=noise)
		finally:
			logger.info(f"# -------- {model} training and evaluation completed -------- # \n \n")
	else:
		logger.info(f"# -------- hyperparameter tuning with {model} - noise {noise} -------- # \n")
		try:
			if model == 'bert':
				exit(1)
			torch.cuda.empty_cache()
			hyperparameter_optimization(model_name=model,
										model_dir=model_dir.format(model=model),
										noisy_triples_file=noisy_triples_file,
										triplets_file_utils=triplets_file_utils,
										ratio=noise)
		finally:
			logger.info(f"# -------- hyperparameter optimization for {model} complete -------- # \n \n")


if __name__ == '__main__':
	main()
	process_results()
