import ast
import sys

import torch
from loguru import logger

from config.config import (model_dir, original_dataset_file, pipeline_config_file, plot_dir,
                           results_dir, triplets_file, valid_kge_models)
from training import hyperparameter_optimization, training
from utils.dataset_utils import generate_noise, generate_triplets, get_train_val_test_factory


def parse_command_line():
	gd = False
	opt = False
	n = False
	if "--generate" in sys.argv:
		gd = ast.literal_eval(sys.argv[sys.argv.index("--generate") + 1])
	if "--optimization" in sys.argv:
		opt = ast.literal_eval(sys.argv[sys.argv.index("--optimization") + 1])
	if "--noise" in sys.argv:
		n = ast.literal_eval(sys.argv[sys.argv.index("--noise") + 1])

	return gd, opt, n


if __name__ == '__main__':

	if len(sys.argv) < 7:
		logger.error("missing arguments from command line")
		raise Exception("missing arguments from command line")


	generate_dataset, optimization, noise = parse_command_line()

	logger.info(f"start, {noise} ratio")

	if generate_dataset:
		generate_triplets(
				original_dataset_file=original_dataset_file,
				triples_file=triplets_file
		)

	train, val, test = generate_noise(triplets_file=triplets_file, noise_ratio=noise)
	train_factory, val_factory, test_factory = get_train_val_test_factory(train, val, test)

	if not optimization:
		logger.info("basic training")

		for model in valid_kge_models:
			torch.cuda.empty_cache()
			training(
					pipeline_config_file=pipeline_config_file,
					model_dir=model_dir.format(model=model),
					model_name=model,
					results_dir=results_dir,
					plot_dir=plot_dir,
					train_factory=train_factory,
					test_factory=test_factory,
					val_factory=val_factory,
					ratio=noise
			)
	else:
		logger.info("hyperparameter tuning")
		for model in valid_kge_models:
			torch.cuda.empty_cache()
			hyperparameter_optimization(model_name=model,
			                            model_dir=model_dir.format(model=model),
			                            train_factory=train_factory,
			                            test_factory=test_factory,
			                            val_factory=val_factory,
			                            ratio=noise)

	logger.info("end")
