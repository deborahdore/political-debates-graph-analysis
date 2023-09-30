import gc
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger
from pykeen.hpo import hpo_pipeline
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from utils.utils import read_json, write_json


def training(
		pipeline_config_file: str,
		model_dir: str,
		model_name: str,
		results_dir: str,
		plot_dir: str,
		train_factory: TriplesFactory,
		test_factory: TriplesFactory,
		val_factory: TriplesFactory,
		ratio: float
) -> None:
	"""
	The training function takes in a pipeline configuration file, a model directory, a model name, and three TriplesFactory
	objects for training data, testing data and validation data.
	The ratio parameter is used to indicate how much noise ratio the dataset in use contains.

	:param pipeline_config_file: str: Read the pipeline configuration file
	:param model_dir: str: Specify the directory where the model is saved
	:param model_name: str: Specify the model to be used
	:param results_dir: str: Save the results to a file
	:param plot_dir: str: Save the loss plot
	:param train_factory: TriplesFactory: Create a training set for the model
	:param test_factory: TriplesFactory: Create the test set
	:param val_factory: TriplesFactory: Create a validation set
	:param ratio: float: Specify the ratio of noise
	"""
	pipeline_config = read_json(pipeline_config_file)

	logger.info(f"starting pipeline --> {model_name} with ratio {ratio}")
	logger.info(f"{pipeline_config}")

	result = pipeline(
			training=train_factory,
			testing=test_factory,
			validation=val_factory,
			model=model_name,
			model_kwargs={'embedding_dim': pipeline_config.get('embedding_dim', 50)},
			optimizer='adam',
			optimizer_kwargs={'lr': pipeline_config.get('learning_rate', 0.01)},
			training_kwargs={'num_epochs':           pipeline_config.get('num_epochs', 5),
			                 'batch_size':           pipeline_config.get('batch_size', 64),
			                 'checkpoint_frequency': 0, },
			use_tqdm=False,
			random_seed=42,
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

	result.save_model(path=model_file)
	gc.collect()


def hyperparameter_optimization(model_name: str,
                                model_dir: str,
                                train_factory: TriplesFactory,
                                test_factory: TriplesFactory,
                                val_factory: TriplesFactory,
                                ratio: float) -> None:
	"""
	The hyperparameter_optimization function takes in a model name, a directory to save the results to, and three
	TriplesFactory objects for training, testing and validation.
	It then runs an HPO pipeline on these data. The number of trials is set at 15.

	:param model_name: str: Specify the model to be used in the pipeline
	:param model_dir:str: Specify the directory where the model will be saved
	:param train_factory: TriplesFactory: Create the training set
	:param test_factory: TriplesFactory: Create a test set
	:param val_factory: TriplesFactory: Create a validation set
	:param ratio:float: Determine the ratio of noise in the dataset
	:return: A directory of the model's results
	:doc-author: Trelent
	"""

	logger.info(f"starting optimizer pipeline - {model_name} with ratio {ratio}")

	hpo_results = hpo_pipeline(
			training=train_factory, testing=test_factory, validation=val_factory, model=model_name,
			n_trials=15, )

	logger.info(f"model {model_name} training complete")

	model_dir_ratio = model_dir + f"/{ratio}"
	logger.info(f"saving {model_name} to {model_dir_ratio}")

	Path(model_dir_ratio).mkdir(exist_ok=True, parents=True)
	hpo_results.save_to_directory(model_dir_ratio)
	gc.collect()
