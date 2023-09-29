import gc

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from pykeen.evaluation import RankBasedEvaluator
from pykeen.losses import MarginRankingLoss
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from utils.utils import read_json, write_json


def get_train_val_test_factory(dataset: str, ratio: float):
	"""
	The get_train_val_test_factory function takes a dataset and a ratio as input.
	It returns the train, validation and test sets for that dataset with the given ratio.
	The returned values are TriplesFactory objects.

	:param dataset: Specify the dataset to use
	:param ratio: Specify the ratio of training data to use
	:return: TriplesFactory objects
	"""
	logger.info("creating train, test and val TriplesFactory")

	train = pd.read_csv(dataset.format(ratio = ratio, use = 'train'), sep = "\t")

	test = pd.read_csv(dataset.format(ratio = ratio, use = 'test'), sep = "\t")

	val = pd.read_csv(dataset.format(ratio = ratio, use = 'val'), sep = "\t")

	train_factory = TriplesFactory.from_labeled_triples(triples = train[['subject', 'predicate', 'object']].values)

	val_factory = TriplesFactory.from_labeled_triples(triples = val[['subject', 'predicate', 'object']].values)

	test_factory = TriplesFactory.from_labeled_triples(triples = test[['subject', 'predicate', 'object']].values)

	return train_factory, val_factory, test_factory


def do_benchmarking(
		model_name: str, dataset_path: str, model_file: str, pipeline_config_file: str, plot_file: str,
		metric_file: str, ratio: float
		):
	"""
	The do_benchmarking function is responsible for running the benchmarking pipeline.

	:param model_name: pykeen.models: Specify the model to be used
	:param dataset_path: str: Specify the path to the dataset
	:param model_file: str: Specify the directory where the trained model will be saved
	:param pipeline_config_file: str: Specify the path to the configuration file for pykeen
	:param plot_file: str: Save the plot of the loss
	:param ratio: float: Split the dataset into train, validation and test sets
	:return: A dictionary with the evaluation metrics
	"""
	pipeline_config = read_json(pipeline_config_file)

	train_factory, val_factory, test_factory = get_train_val_test_factory(dataset_path, ratio)

	logger.info(f"starting pipeline; Model: {model_name}; Ratio: {ratio}")
	logger.info(f"{pipeline_config}")
	result = pipeline(
			model = model_name, random_seed = 42, training = train_factory, testing = test_factory,
			validation = val_factory, model_kwargs = {'embedding_dim': pipeline_config.get('embedding_dim', 50), },
			optimizer = 'adam', optimizer_kwargs = {'lr': pipeline_config.get('learning_rate', 0.01), },
			loss = MarginRankingLoss(), loss_kwargs = {'margin': 1},
			training_kwargs = {'num_epochs':           pipeline_config.get('num_epochs', 5),
			                   'batch_size':           pipeline_config.get('batch_size', 64),
			                   "checkpoint_frequency": 0},
			evaluator = RankBasedEvaluator(),
			# stopper = 'early',
			# stopper_kwargs = dict(frequency = 5, patience = 2, relative_delta = 0.002),
			use_tqdm = False
			)

	logger.info(f"model {model_name} training complete")

	metric_results = {str(key): value for key, value in result.metric_results.data.items()}

	for metric_name in result.metric_results.metrics.keys():
		metric_results[metric_name] = result.get_metric(metric_name)

	write_json(metric_results, metric_file.format(model = model_name, ratio = ratio))

	result.plot_losses()
	plt.savefig(plot_file.format(name = f"{model_name}_loss", ratio = ratio))

	result.save_model(path = model_file.format(model = model_name, ratio = ratio))
	gc.collect()
