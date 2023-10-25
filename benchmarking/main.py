import ast
import sys

import torch
from loguru import logger

from config import config
from config.config import (metrics_file,
						   model_dir,
						   nlp_models,
						   noisy_triples_file,
						   original_dataset_file,
						   plot_dir,
						   triplets_file,
						   triplets_file_utils,
						   valid_kge_models,
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
	if len(sys.argv) < 5:
		logger.error("missing arguments from command line")
		raise Exception("missing arguments from command line")

	assert "--generate" in sys.argv
	assert "--optimize" in sys.argv

	gen_arg = ast.literal_eval(sys.argv[sys.argv.index("--generate") + 1])
	opt_arg = ast.literal_eval(sys.argv[sys.argv.index("--optimize") + 1])

	if opt_arg:
		assert "--model" in sys.argv
		assert "--noise" in sys.argv
		noise_arg = sys.argv[sys.argv.index("--noise") + 1]
		if noise_arg != 'bert':
			noise_arg = ast.literal_eval(noise_arg)
		model_arg = sys.argv[sys.argv.index("--model") + 1]

		return gen_arg, opt_arg, noise_arg, model_arg
	return gen_arg, opt_arg, None, None


if __name__ == '__main__':
	# main()
	process_results()


def main():
	generate_dataset, optimization, noise, model = parse_command_line()

	if generate_dataset:
		generate_triplets(original_dataset_file=original_dataset_file, triples_file=triplets_file)
		generate_mappings(triplets_file=triplets_file, triplets_file_utils=triplets_file_utils)
		generate_noise(triplets_file=triplets_file,
					   noisy_triples_file=noisy_triples_file,
					   valid_noise=config.valid_noise_ratio)

	if not optimization:
		logger.info("basic training with best pipeline config")

		for name in valid_kge_models:
			for noise in valid_noise_ratio:
				try:
					result = training(model_dir=model_dir.format(model=name),
									  model_name=name,
									  noisy_triples_file=noisy_triples_file,
									  triplets_file_utils=triplets_file_utils,
									  plot_dir=plot_dir,
									  ratio=noise)

					torch.cuda.empty_cache()

					link_prediction(result=result,
									noisy_triples_file=noisy_triples_file,
									triplets_file_utils=triplets_file_utils,
									model_name=name,
									metrics_file=metrics_file.format(model=name, ratio=noise),
									noise_ratio=noise)

					torch.cuda.empty_cache()

					link_deletion(result=result,
								  model_name=name,
								  noisy_triples_file=noisy_triples_file,
								  triplets_file_utils=triplets_file_utils,
								  metrics_file=metrics_file.format(model=name, ratio=noise),
								  noise_ratio=noise)

					torch.cuda.empty_cache()

					triple_classification(result=result,
										  model_name=name,
										  noisy_triples_file=noisy_triples_file,
										  triplets_file_utils=triplets_file_utils,
										  metrics_file=metrics_file.format(model=name, ratio=noise),
										  noise_ratio=noise)

				finally:
					logger.info(f"{name} evaluation completed")

		for name in nlp_models:
			for noise in valid_noise_ratio:
				try:
					torch.cuda.empty_cache()

					model = bert_training(model_dir=model_dir.format(model="bert"),
										  model_name='bert',
										  noisy_triples_file=noisy_triples_file,
										  ratio=noise)

					torch.cuda.empty_cache()

					link_prediction_bert(model=model,
										 model_dir=model_dir.format(model="bert"),
										 model_name='bert',
										 noisy_triples_file=noisy_triples_file,
										 metrics_file=metrics_file.format(model=name, ratio=noise),
										 noise_ratio=noise)
					torch.cuda.empty_cache()

					link_deletion_bert(model=model,
									   model_dir=model_dir.format(model="bert"),
									   model_name='bert',
									   noisy_triples_file=noisy_triples_file,
									   metrics_file=metrics_file.format(model=name, ratio=noise),
									   noise_ratio=noise)
					torch.cuda.empty_cache()

					triple_classification_bert(model=model,
											   model_dir=model_dir.format(model="bert"),
											   model_name='bert',
											   noisy_triples_file=noisy_triples_file,
											   metrics_file=metrics_file.format(model=name, ratio=noise),
											   noise_ratio=noise)
				finally:
					logger.info(f"{name} evaluation completed")


	else:
		logger.info(f"hyperparameter tuning with {model} - noise {noise}")
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
		logger.info(f"Training {model} complete")
