import gc
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
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


def bert_training(model_dir: str, model_name: str, noisy_triples_file: str, ratio: float):
	batch_size = 16
	epochs = 5

	logger.info(f"Training bert with {ratio} noise ratio")
	train, val, test = get_train_val_test_from_dir(noisy_triples_file, ratio)
	train_noise, val_noise, test_noise = get_train_val_test_from_dir(noisy_triples_file, 1)

	# benchmarking dataset where we suppose all triples are correct
	train = adjust_dataset_for_bert(train, int(True))
	val = adjust_dataset_for_bert(val, int(True))
	test = adjust_dataset_for_bert(test, int(True))

	# creation of counter examples
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
	model_file = os.path.join(model_dir, f"{ratio}/{model_name}_{ratio}.pt")
	logger.info(f"saving {model_name} to {model_file}")
	torch.save(model, model_file)

	class_report = classification_report(y_true, y_pred)
	logger.info("Classification report")
	logger.info(class_report)

	return model


def training(model_dir: str, model_name: str, noisy_triples_file: str, plot_dir: str, ratio: float):
	pipeline_config = model_dir + f"/{ratio}/best_pipeline/pipeline_config.json"
	pipeline_config = read_json(pipeline_config)['pipeline']

	logger.info(f"starting pipeline --> {model_name} with ratio {ratio} on {device}")
	logger.info(f"{pipeline_config}")

	train, test, val = get_train_val_test_from_dir(noisy_triples_file, noise=ratio)
	train_factory, val_factory, test_factory = get_train_val_test_factory(train, val, test)

	result = pipeline(training=train_factory,
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
					  use_tqdm=True,
					  random_seed=123,
					  device=device)

	model_file = os.path.join(model_dir, f"{ratio}/{model_name}_{ratio}.pt")
	logger.info(f"{model_name} training complete")
	logger.info(f"saving {model_name} to {model_file}")
	result.save_model(path=model_file)

	result.plot_losses()

	plot_file = os.path.join(plot_dir, f"{model_name}_{ratio}_loss.svg")
	plt.savefig(plot_file)
	gc.collect()

	return result


def hyperparameter_optimization(model_name: str, model_dir: str, noisy_triples_file: str, ratio: float):
	logger.info(f"starting optimizer pipeline - {model_name} with ratio {ratio}")

	train, test, val = get_train_val_test_from_dir(noisy_triples_file, noise=ratio)
	train_factory, val_factory, test_factory = get_train_val_test_factory(train, val, test)

	if model_name == 'TransH':
		hpo_results = hpo_pipeline(model=model_name,
								   training=train_factory,
								   testing=test_factory,
								   validation=val_factory,
								   n_trials=15,
								   regularizer=None,
								   optimizer="Adam",
								   optimizer_kwargs_ranges=dict(lr=dict(type=float,
																		low=0.0001,
																		high=0.01,
																		scale="log"), ),
								   training_loop="slcwa",
								   training_kwargs_ranges=dict(num_epochs=dict(type=int, low=30, high=200, q=5),
															   batch_size=dict(type=int, low=64, high=256, q=64), ),
								   negative_sampler="basic",
								   metric="both.realistic.inverse_harmonic_mean_rank",
								   stopper=None,
								   evaluator="RankBasedEvaluator",
								   filter_validation_when_testing=True, )
	else:
		hpo_results = hpo_pipeline(model=model_name,
								   training=train_factory,
								   testing=test_factory,
								   validation=val_factory,
								   n_trials=15,
								   optimizer="Adam",
								   optimizer_kwargs_ranges=dict(lr=dict(type=float,
																		low=0.0001,
																		high=0.01,
																		scale="log"), ),
								   training_loop="slcwa",
								   training_kwargs_ranges=dict(num_epochs=dict(type=int, low=30, high=200, q=5),
															   batch_size=dict(type=int, low=64, high=256, q=64), ),
								   negative_sampler="basic",
								   metric="both.realistic.inverse_harmonic_mean_rank",
								   stopper=None,
								   evaluator="RankBasedEvaluator",
								   filter_validation_when_testing=True, )

	logger.info(f"model {model_name} training complete")

	model_dir_ratio = model_dir + f"/{ratio}"
	logger.info(f"saving {model_name} to {model_dir_ratio}")

	hpo_results.save_to_directory(model_dir_ratio)
	gc.collect()
