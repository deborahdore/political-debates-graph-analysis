import gc
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from optuna.pruners import PercentilePruner
from optuna.samplers import TPESampler
from pykeen.hpo import hpo_pipeline
from pykeen.pipeline import pipeline
from sklearn.metrics import classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup

from utils.dataset_utils import get_train_val_test_factory, get_train_val_test_from_dir
from utils.evaluation_utils import adjust_dataset_for_bert, tokenize_and_generate_dataset
from utils.utils import read_json

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def bert_training(model_file: str, model_name: str, noisy_triples_file: str, ratio: float):
	batch_size = 32
	epochs = 3

	logger.info(f"## ===== BASIC TRAINING {model_name} on {ratio}% noise ratio ===== ##".upper())
	# load dataset
	train, val, test = get_train_val_test_from_dir(noisy_triples_file, ratio, drop_col_noise=True,
												   get_noisy_test=False)

	train = adjust_dataset_for_bert(train, int(True))  # benchmarking dataset where we suppose all triples are correct
	val = adjust_dataset_for_bert(val, int(True))
	test = adjust_dataset_for_bert(test, int(True))

	# creation of counter examples where we have all fake triples
	train_noise, val_noise, test_noise = get_train_val_test_from_dir(noisy_triples_file,
																	 100,
																	 drop_col_noise=False,
																	 get_noisy_test=True)
	train_noise = train_noise[train_noise['noise'] == str(1)].copy()
	val_noise = val_noise[val_noise['noise'] == str(1)].copy()
	test_noise = test_noise[test_noise['noise'] == str(1)].copy()

	train_noise = adjust_dataset_for_bert(train_noise, int(False))
	val_noise = adjust_dataset_for_bert(val_noise, int(False))
	test_noise = adjust_dataset_for_bert(test_noise, int(False))

	train = pd.concat([train, train_noise], axis=0)
	val = pd.concat([val, val_noise], axis=0)
	test = pd.concat([test, test_noise], axis=0)

	dataset_train = tokenize_and_generate_dataset(train)
	dataset_val = tokenize_and_generate_dataset(val)
	dataset_test = tokenize_and_generate_dataset(test)

	# create dataloaders
	dataloader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
	dataloader_val = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)
	dataloader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=batch_size)

	# define model
	model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
														  num_labels=2,
														  output_attentions=False,
														  output_hidden_states=False)
	model.to(device)

	# define optimizer
	optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

	scheduler = get_linear_schedule_with_warmup(optimizer,
												num_warmup_steps=0,
												num_training_steps=len(dataloader_train) * epochs)

	# train
	for epoch in range(epochs):

		loss_train_total = 0.0
		model.train()

		with tqdm(total=len(dataloader_train), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
			for batch in dataloader_train:
				optimizer.zero_grad()

				batch = tuple(b.to(device) for b in batch)
				inputs = {'input_ids': batch[0].long(), 'attention_mask': batch[1], 'labels': batch[2]}
				outputs = model(**inputs)

				loss = outputs.loss
				loss_train_total += loss.item()
				loss.backward()

				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				optimizer.step()
				scheduler.step()

				pbar.update(1)
				pbar.set_postfix({'train/loss': '{:.3f}'.format(loss.item() / len(batch))})

		loss_train_avg = loss_train_total / len(dataloader_train)
		logger.info(f'Training loss: {loss_train_avg}')

		# eval
		model.eval()

		loss_val_total = 0
		with torch.no_grad():
			for batch in dataloader_val:
				batch = tuple(b.to(device) for b in batch)
				inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
				outputs = model(**inputs)

				loss = outputs.loss.float()
				loss_val_total += loss.item()

		loss_val_avg = loss_val_total / len(dataloader_val)
		logger.info(f'Validation loss: {loss_val_avg}')

	logger.info(f"{model_name} training complete")

	# test
	loss_test_total = 0.0
	y_true, y_pred = [], []

	model.eval()

	with tqdm(total=len(dataloader_test), desc="Testing", unit="batch") as pbar:
		for batch in dataloader_test:
			with torch.no_grad():
				batch = tuple(b.to(device) for b in batch)
				inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
				outputs = model(**inputs)
				loss = outputs.loss.float()
				loss_test_total += loss.item()

				res = outputs.logits.detach().cpu().numpy()
				predictions = np.argmax(res, axis=1)
				y_true.extend(batch[2].cpu().numpy())
				y_pred.extend(predictions)

			pbar.update(1)

	avg_test_loss = loss_test_total / len(dataloader_test)
	logger.info(f"[test] Average Test Loss: {avg_test_loss:.20f}")

	# saving model
	logger.info(f"saving {model_name} to {model_file}")
	torch.save(model, model_file)

	class_report = classification_report(y_true, y_pred)
	logger.info("Classification report")
	logger.info(class_report)

	logger.info(f"## ===== BASIC TRAINING COMPLETE ===== ##".upper())

	return model


def training(model_dir: str,
			 model_name: str,
			 model_file: str,
			 noisy_triples_file: str,
			 triplets_file_utils: str,
			 plot_dir: str,
			 ratio: float):
	"""
	The training function is the main function of this module and trains the model with the optimal hyperparameter.

	:param model_file: str: Specify where to save model
	:param model_dir: str: Specify the directory where the model's hyperparameters are saved
	:param model_name: str: Name of the model to train
	:param noisy_triples_file: str: Specify the location of the noisy triples file
	:param triplets_file_utils: str: Contains entity-to-id and relation-to-id mappings
	:param plot_dir: str: Save the plot of the loss function
	:param ratio: float: Determine the amount of noise in the training data
	:return: A result object, which contains the trained model
	"""
	logger.info(f"## ===== BASIC TRAINING {model_name} on {ratio}% noise ratio ===== ##".upper())

	# load datasets
	train, val, test = get_train_val_test_from_dir(noisy_triples_file,
												   noise=ratio,
												   drop_col_noise=False,
												   get_noisy_test=False)
	# creates triples factory
	train_factory, val_factory, test_factory = get_train_val_test_factory(train,
																		  val,
																		  test,
																		  triplets_file_utils,
																		  create_inverse_triples=True)

	# read best hyperparameters
	pipeline_config = os.path.join(model_dir, "best_pipeline/pipeline_config.json")
	pipeline_config = read_json(pipeline_config)['pipeline']
	pipeline_config['training'] = train_factory
	pipeline_config['validation'] = val_factory
	pipeline_config['testing'] = test_factory
	pipeline_config['evaluation_kwargs'] = {}
	pipeline_config['evaluation_kwargs']['additional_filter_triples'] = [train_factory.mapped_triples,
																		 val_factory.mapped_triples]

	result = pipeline(**pipeline_config,
					  device=device,
					  use_testing_data=True,
					  evaluation_fallback=True,
					  use_tqdm=True, )

	# save trained model
	result.save_model(path=model_file)

	# plot losses
	result.plot_losses()
	plot_file = os.path.join(plot_dir, f"{model_name}_{ratio}_loss.svg")
	plt.savefig(plot_file)
	gc.collect()

	logger.info(f"## ===== BASIC TRAINING COMPLETE ===== ##".upper())

	return result


def hyperparameter_optimization(model_name: str, model_dir: str, noisy_triples_file: str, triplets_file_utils: str):
	"""
	The hyperparameter_optimization function is used to optimize the hyperparameters of a model.

	:param model_name: str: Specify the model to be used for training
	:param model_dir: str: Save the model
	:param noisy_triples_file: str: Load the triples from a file
	:param triplets_file_utils: str: Load entity-to-id and relation-to-id mappings
	:param ratio: float: Indicate the noise ratio of the dataset to be used
	"""
	ratio = 0  # only hypertrain on gold

	logger.info(f"## ===== HYPER-OPTIMIZATION TRAINING {model_name} on {ratio}% noise ratio ===== ##".upper())

	# get train, val, test
	train, test, val = get_train_val_test_from_dir(noisy_triples_file, noise=ratio, get_noisy_test=False)
	train_factory, val_factory, test_factory = get_train_val_test_factory(train,
																		  val,
																		  test,
																		  triplets_file_utils,
																		  create_inverse_triples=False)
	# Hyper-training pipeline
	hpo_results = hpo_pipeline(training=train_factory,
							   validation=val_factory,
							   testing=test_factory,
							   model=model_name,
							   optimizer="Adam",
							   optimizer_kwargs_ranges=dict(lr=dict(type=float, low=0.0001, high=0.01, scale="log"), ),
							   training_loop="slcwa",
							   negative_sampler="basic",
							   negative_sampler_kwargs={"filtered": True, "filterer": "python-set", },
							   training_kwargs={"use_tqdm_batch": False, },
							   training_kwargs_ranges=dict(num_epochs=dict(type=int, low=30, high=200, q=5),
														   batch_size=dict(type=int, low=64, high=256, q=64), ),
							   stopper=None,
							   evaluator="RankBasedEvaluator",
							   evaluation_kwargs={
								   "use_tqdm"                 : True,
								   "additional_filter_triples": [train_factory.mapped_triples,
																 val_factory.mapped_triples, ], },
							   evaluator_kwargs={"filtered": True},
							   metric="both.realistic.inverse_harmonic_mean_rank",
							   filter_validation_when_testing=True,
							   device=device,
							   sampler=TPESampler(consider_prior=True,
												  prior_weight=1.0,
												  consider_magic_clip=True,
												  consider_endpoints=False,
												  n_startup_trials=10,
												  n_ei_candidates=24, ),
							   pruner=PercentilePruner(percentile=70.0, n_startup_trials=5, ),
							   direction="maximize",
							   n_trials=10, )

	logger.info(f"## ===== HYPER-OPTIMIZATION TRAINING COMPLETE ===== ##".upper())

	logger.info(f"saving {model_name} to {model_dir}")
	hpo_results.objective.evaluation_kwargs = None
	hpo_results.save_to_directory(model_dir)
	gc.collect()
