import gc
import os.path

import matplotlib.pyplot as plt
from loguru import logger
from pykeen.evaluation import RankBasedEvaluator
from pykeen.hpo import hpo_pipeline
from pykeen.pipeline import pipeline

from utils.dataset_utils import generate_noise, get_train_val_test_factory
from utils.utils import read_json, write_json


def training(
		triplets_file: str,
		pipeline_config_file: str,
		model_dir: str,
		model_name: str,
		results_dir: str,
		plot_dir: str,
		ratio: float
		) -> None:

	"""
	The training function takes in a triplets file, pipeline config file, model directory, model name and results directory.
	It then generates the train/val/test sets using the generate_noise function with a noise ratio of ratio.
	The training factory is created from this train set and passed into the pipeline along with other parameters such as
	the optimizer type and number of epochs to run for. The result is saved to disk as well as some metrics.

	:param triplets_file: str: Specify the file path to the triplets
	:param pipeline_config_file: str: Specify the configuration file for the pipeline
	:param model_dir: str: Specify the directory where the model will be saved
	:param model_name: str: Specify the model to be used
	:param results_dir: str: Store the metric results
	:param plot_dir: str: Save the loss plot
	:param ratio: float: Determine the ratio of noise to be added to the triplets file
	"""
	train, val, test = generate_noise(triplets_file = triplets_file, noise_ratio = ratio)
	train_factory, val_factory, test_factory = get_train_val_test_factory(train, val, test)

	pipeline_config = read_json(pipeline_config_file)

	logger.info(f"starting pipeline --> {model_name} with ratio {ratio}")
	logger.info(f"{pipeline_config}")

	result = pipeline(
			training = train_factory,
			testing = test_factory,
			validation = val_factory,
			model = model_name,
			model_kwargs = {'embedding_dim': pipeline_config.get('embedding_dim', 50)},
			optimizer = 'adam',
			optimizer_kwargs = {'lr': pipeline_config.get('learning_rate', 0.01)},
			training_kwargs = {'num_epochs': pipeline_config.get('num_epochs', 5),
			                   'batch_size': pipeline_config.get('batch_size', 64),
			                   'checkpoint_frequency': 0, },
			use_tqdm = False,
			random_seed = 42,
			)

	model_file = os.path.join(model_dir, f"{model_name}_{ratio}.pt")

	logger.info(f"{model_name} training complete")
	logger.info(f"saving {model_name} to {model_file}")

	metric_results = {str(key): value for key, value in result.metric_results.data.items()}

	for metric_name in result.metric_results.metrics.keys():
		metric_results[metric_name] = result.get_metric(metric_name)

	metric_file = os.path.join(results_dir, f"metric_{model_name}_{ratio}.json")
	write_json(metric_results, metric_file)

	result.plot_losses()

	plot_file = os.path.join(plot_dir, f"{model_name}_{ratio}_loss.svg")
	plt.savefig(plot_file)

	result.save_model(path = model_file)
	gc.collect()


def hyperparameter_optimization(triplets_file:str, model_name: str, model_dir:str, ratio:float) -> None:
	"""
	The hyperparameter_optimization function takes in a triplets file, model name, model directory and ratio.
	It then generates noise for the train, val and test sets using the generate_noise function. It then gets
	the training factory, validation factory and testing factory from get_train_val_test_factory function.
	The hpo pipeline is run with 15 trials on these factories to optimize hyperparameters for each of them.
	The results are saved to a directory.

	:param triplets_file:str: Specify the file containing the triplets
	:param model_name: str: Specify the model to be used for training
	:param model_dir:str: Specify the directory where the model will be saved
	:param ratio:float: Specify the noise ratio for the triplets
	"""
	train, val, test = generate_noise(triplets_file = triplets_file, noise_ratio = ratio)
	train_factory, val_factory, test_factory = get_train_val_test_factory(train, val, test)

	logger.info(f"starting optimizer pipeline; Model: {model_name}; Ratio: {ratio}")

	hpo_results = hpo_pipeline(
			training = train_factory, testing = test_factory, validation = val_factory, model = model_name,
			n_trials = 15, )

	logger.info(f"model {model_name} training complete")

	model_file = os.path.join(model_dir, f"hyper/{model_name}_{ratio}")
	logger.info(f"saving {model_name} to {model_file}")

	hpo_results.save_to_directory(model_dir)
	gc.collect()