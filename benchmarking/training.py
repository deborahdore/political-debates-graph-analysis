import os.path

import numpy as np
import torch
from loguru import logger
from optuna.pruners import PercentilePruner
from optuna.samplers import TPESampler
from pykeen.hpo import hpo_pipeline
from pykeen.nn.init import PretrainedInitializer
from pykeen.pipeline import pipeline

from benchmarking import config
from benchmarking.utils.dataset_utils import get_train_val_test_factory, get_train_val_test_from_dir
from benchmarking.utils.util import load_model, read_json


def training(model_name: str,
			 model_dir: str,
			 noisy_split_triplets_file: str,
			 triplets_file_utils: str,
			 task_name: str,
			 ratio: float,
			 pretrained_embedding_file: str = None):
	""" Train a KGE model """

	model_file = f"{model_dir.format(model=model_name, task_name=task_name)}/{model_name}.pt"
	if os.path.exists(model_file):
		return load_model(model_file)

	logger.info(f"🚀 Basic training {model_name} with {ratio}% noise")

	train, test, val = get_train_val_test_from_dir(noisy_split_triplets_file,
												   noise=ratio,
												   drop_col_noise=False,
												   get_noisy_test=False)
	logger.info("📊 Relation counts: ")
	logger.info(train['relation'].value_counts())

	train_factory, val_factory, test_factory = get_train_val_test_factory(train,
																		  val,
																		  test,
																		  triplets_file_utils,
																		  create_inverse_triples=True)

	pipeline_config = f"{model_dir.format(model=model_name, task_name=task_name)}/best_pipeline/pipeline_config.json"
	assert os.path.isfile(pipeline_config)

	pipeline_config = read_json(pipeline_config)['pipeline']

	if config.USE_PRETRAINED_EMBEDDINGS:
		logger.info("🚀 Training with pretrained embeddings")
		assert pretrained_embedding_file is not None
		pretrained_embedding_tensor = torch.FloatTensor(np.load(pretrained_embedding_file)).to(config.DEVICE)
		pipeline_config['model_kwargs'] = dict(embedding_dim=pretrained_embedding_tensor.shape[-1],
											   entity_initializer=PretrainedInitializer(
												   tensor=pretrained_embedding_tensor))
	if config.SPECIAL_BENCHMARKING_FLAG:
		logger.info("🚀 Training with only relevant relations")
		pipeline_config['evaluation_relation_whitelist'] = ['__label__Support', '__label__Attack',
															'__label__Equivalent']

	if config.WANDB:
		pipeline_config['result_tracker'] = "wandb"
		pipeline_config['result_tracker_kwargs'] = dict(project=config.WANDB_PROJECT_NAME)
		pipeline_config['metadata'] = dict(title=f'{task_name}-{model_name}-noise={ratio}')

	logger.info(f"📊 Best params: {pipeline_config}")

	pipeline_config['training'] = train_factory
	pipeline_config['validation'] = val_factory
	pipeline_config['testing'] = test_factory
	pipeline_config['evaluation_kwargs']['additional_filter_triples'] = [train_factory.mapped_triples,
																		 val_factory.mapped_triples, ]
	pipeline_config['device'] = config.DEVICE
	pipeline_config['use_testing_data'] = True
	pipeline_config['evaluation_fallback'] = True
	pipeline_config['filter_validation_when_testing'] = True
	pipeline_config['use_tqdm'] = True
	pipeline_config['random_seed'] = config.SEED

	result = pipeline(**pipeline_config)

	result.save_model(path=model_file)

	logger.info("🚀 Basic training complete")
	return result.model


def optimization(model_name: str,
				 model_dir: str,
				 noisy_split_triplets_file: str,
				 triplets_file_utils: str,
				 task_name: str,
				 pretrained_embedding_file: str = None):
	"""	Optimize the hyperparameters of a model	"""

	logger.info(f"🚀 Hyper-optimization {model_name}")

	train, test, val = get_train_val_test_from_dir(noisy_split_triplets_file, noise=0, get_noisy_test=False)

	logger.info("📊 Relation counts: ")
	logger.info(train['relation'].value_counts())

	train_factory, val_factory, test_factory = get_train_val_test_factory(train,
																		  val,
																		  test,
																		  triplets_file_utils,
																		  create_inverse_triples=False)

	if config.USE_PRETRAINED_EMBEDDINGS:
		logger.info("🚀 Training with pretrained embeddings")
		assert pretrained_embedding_file is not None
		pretrained_embedding_tensor = torch.FloatTensor(np.load(pretrained_embedding_file)).to(config.DEVICE)
		model_kwargs = dict(embedding_dim=pretrained_embedding_tensor.shape[-1],
							entity_initializer=PretrainedInitializer(tensor=pretrained_embedding_tensor))
	else:
		model_kwargs = None

	evaluation_relation_whitelist = None
	if config.SPECIAL_BENCHMARKING_FLAG:
		logger.info("🚀 Training with only relevant relations")
		evaluation_relation_whitelist = ['__label__Support', '__label__Attack', '__label__Equivalent']

	hpo_results = hpo_pipeline(  # 1. Dataset
		training=train_factory,
		validation=val_factory,
		testing=test_factory,
		evaluation_relation_whitelist=evaluation_relation_whitelist,
		# 2. Model
		model=model_name,
		model_kwargs=model_kwargs,
		# 5. Optimizer
		optimizer="Adam",
		# 6. Training Loop
		training_loop="slcwa",
		negative_sampler="basic",
		negative_sampler_kwargs=dict(filterer='python-set', filtered=True),
		# 7. Training
		training_kwargs=dict(use_tqdm_batch=False),
		training_kwargs_ranges=dict(num_epochs=dict(type=int, low=30, high=200, q=5),
									batch_size=dict(type=int, low=64, high=256, q=64), ),
		# 8. Evaluation
		evaluator="RankBasedEvaluator",
		evaluation_kwargs={
			"use_tqdm"                 : True,
			"additional_filter_triples": [train_factory.mapped_triples, val_factory.mapped_triples, ], },
		evaluator_kwargs={"filtered": True, },
		metric="both.realistic.inverse_harmonic_mean_rank",
		filter_validation_when_testing=True,
		# 6. Misc
		device=config.DEVICE,
		#  Optuna Study Settings
		sampler=TPESampler(consider_prior=True,
						   prior_weight=1.0,
						   consider_magic_clip=True,
						   consider_endpoints=False,
						   n_startup_trials=18,
						   n_ei_candidates=32, ),
		pruner=PercentilePruner(percentile=70.0, n_startup_trials=5, ),
		direction="maximize",
		n_trials=config.NUM_TRIALS, )

	logger.info(f"🚀 Best hyper-parameters: {hpo_results.study.best_params}")
	logger.info("🚀 Hyper-optimization complete")

	hpo_results.objective.evaluation_kwargs['additional_filter_triples'] = None
	hpo_results.objective.model_kwargs = None
	hpo_results.save_to_directory(model_dir.format(model=model_name, task_name=task_name))
