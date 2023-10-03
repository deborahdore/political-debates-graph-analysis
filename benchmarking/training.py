import gc
import os.path
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from loguru import logger
from pykeen.hpo import hpo_pipeline
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from utils.utils import read_json


def training(model_dir: str,
             model_name: str,
             plot_dir: str,
             train_factory: TriplesFactory,
             test_factory: TriplesFactory,
             val_factory: TriplesFactory,
             ratio: float) -> Any:
	"""
	The training function takes in a pipeline configuration file, a model directory, a model name, and three
	TriplesFactory
	objects for training data, testing data and validation data.
	The ratio parameter is used to indicate how much noise ratio the dataset in use contains.

	:param model_dir: str: Specify the directory where the model is saved
	:param model_name: str: Specify the model to be used
	:param results_dir: str: Save the results to a file
	:param plot_dir: str: Save the loss plot
	:param train_factory: TriplesFactory: Create a training set for the model
	:param test_factory: TriplesFactory: Create the test set
	:param val_factory: TriplesFactory: Create a validation set
	:param ratio: float: Specify the ratio of noise
	"""
	pipeline_config = model_dir + f"/{ratio}/best_pipeline/pipeline_config.json"
	pipeline_config = read_json( pipeline_config )['pipeline']

	logger.info( f"starting pipeline --> {model_name} with ratio {ratio}" )
	logger.info( f"{pipeline_config}" )

	result = pipeline( training=train_factory,
	                   testing=test_factory,
	                   validation=val_factory,
	                   model=model_name,
	                   evaluator=pipeline_config['evaluator'],
	                   filter_validation_when_testing=pipeline_config['filter_validation_when_testing'],
	                   loss=pipeline_config['loss'],
	                   loss_kwargs=pipeline_config['loss_kwargs'],
	                   model_kwargs=pipeline_config['model_kwargs'],
	                   negative_sampler=pipeline_config['negative_sampler'],
	                   negative_sampler_kwargs=pipeline_config['negative_sampler_kwargs'],
	                   optimizer=pipeline_config['optimizer'],
	                   optimizer_kwargs=pipeline_config['optimizer_kwargs'],
	                   training_kwargs=pipeline_config['training_kwargs'],
	                   training_loop=pipeline_config['training_loop'],
	                   use_tqdm=False,
	                   random_seed=42, )

	model_file = os.path.join( model_dir, f"{model_name}_{ratio}.pt" )

	logger.info( f"{model_name} training complete" )
	logger.info( f"saving {model_name} to {model_file}" )
	result.save_model( path=model_file )

	result.plot_losses()

	plot_file = os.path.join( plot_dir, f"{model_name}_{ratio}_loss.svg" )
	plt.savefig( plot_file )
	gc.collect()
	return result


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

	logger.info( f"starting optimizer pipeline - {model_name} with ratio {ratio}" )

	hpo_results = hpo_pipeline( training=train_factory,
	                            testing=test_factory,
	                            validation=val_factory,
	                            model=model_name,
	                            n_trials=15,
	                            evaluator="RankBasedEvaluator",
	                            )

	logger.info( f"model {model_name} training complete" )

	model_dir_ratio = model_dir + f"/{ratio}"
	logger.info( f"saving {model_name} to {model_dir_ratio}" )

	Path( model_dir_ratio ).mkdir( exist_ok=True, parents=True )
	hpo_results.save_to_directory( model_dir_ratio )
	gc.collect()
