import ast
import sys

import torch
from loguru import logger

from config.config import (metrics_file, model_dir, original_dataset_file, plot_dir, triplets_file)
from evaluation import link_deletion_evaluation, link_prediction_evaluation
from training import hyperparameter_optimization, training
from utils.dataset_utils import generate_noise, generate_triplets, get_train_val_test_factory


def parse_command_line():
	assert "--noise" in sys.argv
	noise_arg = ast.literal_eval(sys.argv[sys.argv.index("--noise") + 1])

	gen_arg = False
	if "--generate" in sys.argv:
		gen_arg = ast.literal_eval(sys.argv[sys.argv.index("--generate") + 1])

	opt_arg = False
	if "--optimization" in sys.argv:
		assert "--model" in sys.argv
		opt_arg = ast.literal_eval(sys.argv[sys.argv.index("--optimization") + 1])
		model_arg = sys.argv[sys.argv.index("--model") + 1]
		return gen_arg, opt_arg, noise_arg, model_arg

	return gen_arg, opt_arg, noise_arg, None


if __name__ == '__main__':

	if len(sys.argv) < 5:
		logger.error("missing arguments from command line")
		raise Exception("missing arguments from command line")

	generate_dataset, optimization, noise, model = parse_command_line()

	logger.info(f"start, {noise} ratio")

	if generate_dataset:
		generate_triplets(original_dataset_file=original_dataset_file, triples_file=triplets_file)

	train, val, test = generate_noise(triplets_file=triplets_file, noise_ratio=noise)
	train_factory, val_factory, test_factory = get_train_val_test_factory(train, val, test)

	if not optimization:
		logger.info("basic training with best pipeline config")

		for name in ['TransE']:
			torch.cuda.empty_cache()
			result = training(model_dir=model_dir.format(model=name),
							  model_name=name,
							  plot_dir=plot_dir,
							  train_factory=train_factory,
							  test_factory=test_factory,
							  val_factory=val_factory,
							  ratio=noise)

			link_deletion_evaluation(result=result,
									 model_name=name,
									 triples_file=triplets_file.format(use="test"),
									 metrics_file=metrics_file.format(model=name, ratio=noise),
									 ratio=noise)

			link_prediction_evaluation(result=result,
									   noisy_train=train_factory,
									   noisy_val=val_factory,
									   original_test_file=triplets_file.format(use="test"),
									   model_name=name,
									   metrics_file=metrics_file.format(model=name, ratio=noise),
									   ratio=noise)

	else:
		logger.info(f"hyperparameter tuning with {model} - noise {noise}")
	try:
		torch.cuda.empty_cache()
		hyperparameter_optimization(model_name=model,
									model_dir=model_dir.format(model=model),
									train_factory=train_factory,
									test_factory=test_factory,
									val_factory=val_factory,
									ratio=noise)
	finally:
		logger.info(f"Training {model} complete")

logger.info("end")
