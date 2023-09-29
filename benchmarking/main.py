import ast
import sys

import torch
from loguru import logger

from config.config import (model_dir, original_dataset_file, pipeline_config_file, plot_dir, ratio,
                           results_dir, triplets_file, valid_kge_models)
from training import hyperparameter_optimization, training
from utils.dataset_utils import generate_triplets


def parse_command_line():
	gd = False
	opt = False
	if "--generate" in sys.argv:
		gd = ast.literal_eval(sys.argv[sys.argv.index("--generate") + 1])
	if "--optimization" in sys.argv:
		opt = ast.literal_eval(sys.argv[sys.argv.index("--optimization") + 1])

	return gd, opt


if __name__ == '__main__':
	logger.info("start")

	if len(sys.argv) < 5:
		logger.error("missing arguments from command line")
		raise Exception("missing arguments from command line")

	generate_dataset, optimization = parse_command_line()

	if generate_dataset:
		generate_triplets(
				original_dataset_file = original_dataset_file,
				triples_file = triplets_file
			)

	if not optimization:
		logger.info("basic training")

		for model in valid_kge_models:
			for noise in ratio:
				torch.cuda.empty_cache()
				training(
						triplets_file = triplets_file,
						pipeline_config_file = pipeline_config_file,
						model_dir = model_dir.format(model=model),
						model_name = model,
						results_dir = results_dir,
						plot_dir = plot_dir,
						ratio = noise
						)
	else:
		logger.info("hyperparameter tuning")
		for model in valid_kge_models:
			for noise in ratio:
				torch.cuda.empty_cache()
				hyperparameter_optimization(triplets_file = triplets_file,
				                            model_name = model,
				                            model_dir = model_dir.format(model=model),
				                            ratio = noise)

	logger.info("end")
