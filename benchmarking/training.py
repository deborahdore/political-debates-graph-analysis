import gc
import os.path

import numpy as np
import torch
from loguru import logger
from optuna.pruners import PercentilePruner
from optuna.samplers import TPESampler
from pykeen.hpo import hpo_pipeline
from pykeen.nn.init import PretrainedInitializer
from pykeen.pipeline import pipeline

import config
from utils.dataset_utils import get_train_val_test_factory, get_train_val_test_from_dir
from utils.utils import read_json


def training(model_dir: str,
			 model_name: str,
			 model_file: str,
			 noisy_triples_file: str,
			 triplets_file_utils: str,
			 ratio: float,
			 pretrained_embedding_file: str = None):
	"""
	Train KGE model

	:param model_file: str: file where the model will be saved
	:param model_dir: str: directory where the model's hyperparameters are saved
	:param model_name: str: name of the model to train
	:param noisy_triples_file: str: dataset file
	:param triplets_file_utils: str: file where entity-to-id and relation-to-id mappings are stored
	:param pretrained_embedding_file: file that contains pretrained Embeddings
	:param ratio: float: the amount of noise in the training data

	:return: A result object, which contains the trained model
	"""
	logger.info(f"## ===== BASIC TRAINING {model_name} on {ratio}% noise ratio ===== ##".upper())

	# load datasets
	train, val, test = get_train_val_test_from_dir(noisy_triples_file,
												   noise=ratio,
												   drop_col_noise=False,
												   get_noisy_test=False)
	logger.info("RELATION COUNTS: ")
	logger.info(train['relation'].value_counts())

	# creates triples factory
	train_factory, val_factory, test_factory = get_train_val_test_factory(train,
																		  val,
																		  test,
																		  triplets_file_utils,
																		  create_inverse_triples=True)

	# read best hyperparameters
	pipeline_config = os.path.join(model_dir, "best_pipeline/pipeline_config.json")
	assert os.path.isfile(pipeline_config)

	pipeline_config = read_json(pipeline_config)['pipeline']

	# check negative_sampler_kwargs
	if not 'negative_sampler_kwargs' in pipeline_config.keys():
		pipeline_config['negative_sampler_kwargs'] = {}
	pipeline_config['negative_sampler_kwargs']['filtered'] = True
	pipeline_config['negative_sampler_kwargs']["filterer"] = 'python-set'

	# check training_kwargs
	if not 'training_kwargs' in pipeline_config.keys():
		pipeline_config['training_kwargs'] = {"num_epochs": 100, "batch_size": 128, "use_tqdm_batch": False, }

	# check regularizer_kwargs
	if not 'regularizer_kwargs' in pipeline_config.keys():
		pipeline_config['regularizer_kwargs'] = None

	#  check loss_kwargs
	if not 'loss_kwargs' in pipeline_config.keys():
		pipeline_config['loss_kwargs'] = None

	if config.USE_PRETRAINED_EMBEDDINGS:
		assert pretrained_embedding_file is not None
		pretrained_embedding_tensor = torch.FloatTensor(np.load(pretrained_embedding_file)).to(config.DEVICE)
		pipeline_config['model_kwargs'] = dict(embedding_dim=pretrained_embedding_tensor.shape[-1],
											   entity_initializer=PretrainedInitializer(
												   tensor=pretrained_embedding_tensor))

	logger.info(f"Best params: {pipeline_config}")

	result = pipeline(  # dataset args
		training=train_factory,
		validation=val_factory,
		testing=test_factory,
		result_tracker='wandb' if config.WANDB else None,
		result_tracker_kwargs=dict(project=config.WANDB_PROJECT_NAME) if config.WANDB else None,
		metadata=dict(title=f'{config.base_dir.split("/")[-1]}_{model_name}'),
		# model args
		model=model_name,
		model_kwargs=pipeline_config['model_kwargs'],
		# loss args
		loss_kwargs=pipeline_config['loss_kwargs'],
		# regularize args
		regularizer_kwargs=pipeline_config['regularizer_kwargs'],
		# optimizer args
		optimizer='Adam',
		optimizer_kwargs=pipeline_config['optimizer_kwargs'],
		clear_optimizer=True,
		# training Loop args
		training_loop='slcwa',
		negative_sampler="basic",
		negative_sampler_kwargs=dict(filterer='python-set',
									 filtered=True,
									 corruption_scheme=('head', 'relation', 'tail')),
		# training args
		training_kwargs=pipeline_config['training_kwargs'],
		stopper=None,
		# evaluation args
		evaluator="RankBasedEvaluator",
		evaluator_kwargs={"filtered": True, },
		evaluation_kwargs={
			"use_tqdm"                 : True,
			"additional_filter_triples": [train_factory.mapped_triples, val_factory.mapped_triples, ], },
		use_testing_data=True,
		evaluation_fallback=True,
		filter_validation_when_testing=True,
		use_tqdm=True,
		device=config.DEVICE,
		random_seed=123)

	# save trained model
	result.save_model(path=model_file)

	gc.collect()
	logger.info(f"## ===== BASIC TRAINING COMPLETE ===== ##".upper())

	return result


def hyperparameter_optimization(model_name: str,
								model_dir: str,
								noisy_triples_file: str,
								triplets_file_utils: str,
								pretrained_embedding_file: str = None):
	"""
	Optimize the hyperparameters of a model

	:param model_name: str: name of the model to optimize
	:param model_dir: str: directory where the best hyperparameters will be saved
	:param noisy_triples_file: str: dataset file
	:param triplets_file_utils: str: file where entity-to-id and relation-to-id mappings are stored
	:param pretrained_embedding_file: str: file where the pretrained embeddings are saved
	"""

	logger.info(f"## ===== HYPER-OPTIMIZATION TRAINING {model_name} ===== ##".upper())

	# get train, val, test
	train, test, val = get_train_val_test_from_dir(noisy_triples_file, noise=0, get_noisy_test=False)

	logger.info("RELATION COUNTS: ")
	logger.info(train['relation'].value_counts())

	train_factory, val_factory, test_factory = get_train_val_test_factory(train,
																		  val,
																		  test,
																		  triplets_file_utils,
																		  create_inverse_triples=False)

	if config.USE_PRETRAINED_EMBEDDINGS:
		assert pretrained_embedding_file is not None
		pretrained_embedding_tensor = torch.FloatTensor(np.load(pretrained_embedding_file)).to(config.DEVICE)
		model_kwargs = dict(embedding_dim=pretrained_embedding_tensor.shape[-1],
							entity_initializer=PretrainedInitializer(tensor=pretrained_embedding_tensor))
	else:
		model_kwargs = None

	hpo_results = hpo_pipeline(training=train_factory,
							   validation=val_factory,
							   testing=test_factory,
							   model=model_name,
							   model_kwargs=model_kwargs,
							   # optimizer args
							   optimizer="Adam",
							   # training loop args
							   training_loop="slcwa",
							   negative_sampler="basic",
							   negative_sampler_kwargs=dict(filterer='python-set',
															filtered=True,
															corruption_scheme=('head', 'relation', 'tail')),
							   # training args
							   training_kwargs=dict(use_tqdm_batch=False),
							   training_kwargs_ranges=dict(num_epochs=dict(type=int, low=30, high=200, q=5),
														   batch_size=dict(type=int, low=64, high=256, q=64), ),
							   stopper=None,
							   # evaluation args
							   evaluator="RankBasedEvaluator",
							   evaluation_kwargs={
								   "use_tqdm"                 : True,
								   "additional_filter_triples": [train_factory.mapped_triples,
																 val_factory.mapped_triples, ], },
							   evaluator_kwargs={"filtered": True, },
							   metric="both.realistic.inverse_harmonic_mean_rank",
							   # MRR
							   filter_validation_when_testing=True,
							   # misc args
							   device=config.DEVICE,
							   # Optuna study args
							   sampler=TPESampler(consider_prior=True,
												  prior_weight=1.0,
												  consider_magic_clip=True,
												  consider_endpoints=False,
												  n_startup_trials=18,
												  n_ei_candidates=32, ),
							   pruner=PercentilePruner(percentile=70.0, n_startup_trials=5, ),
							   direction="maximize",
							   n_trials=config.NUM_TRIALS, )

	logger.info(f"Best hyper-parameters: {hpo_results.study.best_params}")
	logger.info(f"## ===== HYPER-OPTIMIZATION TRAINING COMPLETE ===== ##".upper())

	logger.info(f"saving {model_name} to {model_dir}")
	hpo_results.objective.evaluation_kwargs['additional_filter_triples'] = None
	hpo_results.objective.model_kwargs = None
	hpo_results.save_to_directory(model_dir)
	gc.collect()
