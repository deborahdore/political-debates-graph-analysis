import os
import random

import numpy as np
import pandas as pd
import torch
from loguru import logger
from pykeen.evaluation import RankBasedEvaluator
from pykeen.metrics.ranking import AdjustedArithmeticMeanRank, \
	AdjustedInverseHarmonicMeanRank, \
	ArithmeticMeanRank, \
	HitsAtK, \
	InverseHarmonicMeanRank
from pykeen.pipeline import PipelineResult
from sklearn import metrics
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertForSequenceClassification

from utils.dataset_utils import get_factory, get_nodes, get_train_val_test_factory, get_train_val_test_from_dir
from utils.evaluation_utils import adjust_dataset_for_bert, \
	get_center, \
	get_probabilities_bert, \
	get_scores, \
	tokenize_and_generate_dataset
from utils.utils import read_json, save_json

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# ======================== KGE MODELS EVALUATION  ======================== #
def link_deletion_evaluation(result: PipelineResult,
							 model_name: str,
							 noisy_triples_file: str,
							 metrics_file: str,
							 noise_ratio: float):
	# load pytorch model
	model = result.model

	# load dataset
	train_original, val_original, test_original = get_train_val_test_from_dir(noisy_triples_file, 0)

	# get factory
	train_factory, val_factory, test_factory = get_train_val_test_factory(train_original,
																		  val_original,
																		  test_original,
																		  False)

	logger.info(f"evaluating {model_name} trained with {noise_ratio} noise ratio dataset on link deletion")

	real_test_scores = get_scores(model, test_factory, result.training)

	ranks = []
	ranks_head = []
	ranks_tail = []

	mr_calculator = ArithmeticMeanRank()
	adjusted_mr_calculator = AdjustedArithmeticMeanRank()
	mrr_calculator = InverseHarmonicMeanRank()
	adjusted_mrr_calculator = AdjustedInverseHarmonicMeanRank()
	hits_at_1_calculator = HitsAtK(k=1)
	hits_at_3_calculator = HitsAtK(k=3)
	hits_at_5_calculator = HitsAtK(k=5)
	hits_at_10_calculator = HitsAtK(k=10)

	dataset_original = pd.concat([train_original, val_original, test_original], axis=0).reset_index(drop=True)
	nodes = get_nodes(dataset_original)

	for h, r, t in test_original.values.tolist():
		fake_head_triple = None
		fake_tail_triple = None

		new_h = random.choice(nodes)
		# assert not real triple
		while [new_h, r, t] in dataset_original.values.tolist():
			new_h = random.choice(nodes)
		fake_head_triple = [new_h, r, t]

		new_t = random.choice(nodes)
		# assert not real triple
		while [h, r, new_t] in dataset_original.values.tolist():
			new_t = random.choice(nodes)
		fake_tail_triple = [h, r, new_t]

		fake_factory = get_factory(pd.DataFrame([fake_head_triple, fake_tail_triple], columns=train_original.columns))
		scores = get_scores(model, fake_factory, result.training)

		assert len(scores) == 2
		fake_h_score, fake_t_score = scores[0], scores[1]
		rank_head = np.searchsorted(a=real_test_scores, v=fake_h_score, side='left') + 1
		rank_tail = np.searchsorted(a=real_test_scores, v=fake_t_score, side='left') + 1
		ranks.append(int((rank_head + rank_tail) / 2.0))
		ranks_head.append(rank_head)
		ranks_tail.append(rank_tail)

	# Metrics
	test_size = len(test_original)
	n_round = 4

	ranks_head_array = np.array(ranks_head, dtype=int)
	ranks_tail_array = np.array(ranks_tail, dtype=int)

	# MR
	mr_head = round(float(mr_calculator(ranks_head_array, test_size)), n_round)
	mr_tail = round(float(mr_calculator(ranks_tail_array, test_size)), n_round)
	mr = round(int((mr_head + mr_tail) / 2.0), n_round)

	# ADJUSTED MR
	adjusted_mr_head = round(float(adjusted_mr_calculator(ranks_head_array, test_size)), n_round)
	adjusted_mr_tail = round(float(adjusted_mr_calculator(ranks_tail_array, test_size)), n_round)
	adjusted_mr = round(int((mr_head + mr_tail) / 2.0), n_round)

	# MRR
	mrr_head = round(float(mrr_calculator(ranks_head_array, test_size)), n_round)
	mrr_tail = round(float(mrr_calculator(ranks_tail_array, test_size)), n_round)
	mrr = round(float((mrr_head + mrr_tail) / 2.0), n_round)

	# ADJUSTED MRR
	adjusted_mrr_head = round(float(adjusted_mrr_calculator(ranks_head_array, test_size)), n_round)
	adjusted_mrr_tail = round(float(adjusted_mrr_calculator(ranks_tail_array, test_size)), n_round)
	adjusted_mrr = round(float((mrr_head + mrr_tail) / 2.0), n_round)

	# HITS AT 1
	hits_at_1_head = round(float(hits_at_1_calculator(ranks_head_array)), n_round)
	hits_at_1_tail = round(float(hits_at_1_calculator(ranks_tail_array)), n_round)
	hits_at_1 = round(float((hits_at_1_head + hits_at_1_tail) / 2.0), n_round)

	# HITS AT 3
	hits_at_3_head = round(float(hits_at_3_calculator(ranks_head_array)), n_round)
	hits_at_3_tail = round(float(hits_at_3_calculator(ranks_tail_array)), n_round)
	hits_at_3 = round(float((hits_at_3_head + hits_at_3_tail) / 2.0), n_round)

	# HITS AT 5
	hits_at_5_head = round(float(hits_at_5_calculator(ranks_head_array)), n_round)
	hits_at_5_tail = round(float(hits_at_5_calculator(ranks_tail_array)), n_round)
	hits_at_5 = round(float((hits_at_5_head + hits_at_5_tail) / 2.0), n_round)

	# HITS AT 10
	hits_at_10_head = round(float(hits_at_10_calculator(ranks_head_array)), n_round)
	hits_at_10_tail = round(float(hits_at_10_calculator(ranks_tail_array)), n_round)
	hits_at_10 = round(float((hits_at_10_head + hits_at_10_tail) / 2.0), n_round)

	results_eval = {
		'head': {
			'hits_at_1'   : hits_at_1_head,
			'hits_at_3'   : hits_at_3_head,
			'hits_at_5'   : hits_at_5_head,
			'hits_at_10'  : hits_at_10_head,
			'mr'          : mr_head,
			'adjusted_mr' : adjusted_mr_head,
			'mrr'         : mrr_head,
			'adjusted_mrr': adjusted_mrr_head},
		'both': {
			'hits_at_1'   : hits_at_1,
			'hits_at_3'   : hits_at_3,
			'hits_at_5'   : hits_at_5,
			'hits_at_10'  : hits_at_10,
			'mr'          : mr,
			'adjusted_mr' : adjusted_mr,
			'mrr'         : mrr,
			'adjusted_mrr': adjusted_mrr},
		'tail': {
			'hits_at_1'   : hits_at_1_tail,
			'hits_at_3'   : hits_at_3_tail,
			'hits_at_5'   : hits_at_5_tail,
			'hits_at_10'  : hits_at_10_tail,
			'mr'          : mr_tail,
			'adjusted_mr' : adjusted_mr_tail,
			'mrr'         : mrr_tail,
			'adjusted_mrr': adjusted_mrr_tail},

	}

	# Check if the JSON file exists
	if os.path.isfile(metrics_file):
		existing_results = read_json(metrics_file)
		existing_results.update({"link deletion": results_eval})
		save_json(existing_results, metrics_file)
	else:
		save_json({"link deletion": results_eval}, metrics_file)

	logger.info(f"Evaluating model {model_name} complete")


def link_prediction_evaluation(result: PipelineResult,
							   noisy_triples_file: str,
							   model_name: str,
							   metrics_file: str,
							   noise_ratio: float):
	logger.info(f"evaluating {model_name} trained with {noise_ratio} noise ratio dataset on link prediction")

	evaluator = RankBasedEvaluator(filtered=True)

	# load dataset
	train_original, val_original, test_original = get_train_val_test_from_dir(noisy_triples_file, 0, False)
	train_noisy, val_noisy, test_noisy = get_train_val_test_from_dir(noisy_triples_file, noise_ratio, False)

	train_factory, val_factory, test_factory = get_train_val_test_factory(train_original,
																		  val_original,
																		  test_original,
																		  False)
	train_factory_noisy, val_factory_noisy, test_factory_noisy = get_train_val_test_factory(train_noisy,
																							val_noisy,
																							test_noisy,
																							False)

	result_dict = evaluator.evaluate(model=result.model,
									 mapped_triples=test_factory.mapped_triples,
									 batch_size=result.configuration.get('batch_size'),
									 additional_filter_triples=[train_factory_noisy.mapped_triples,
																val_factory_noisy.mapped_triples],
									 use_tqdm=True,
									 slice_size=None,
									 device=torch.device(device)).to_dict()

	results_eval = {}
	n_round = 4
	for key in ['head', 'both', 'tail']:
		sub_dict = result_dict[key]['realistic']
		results_eval[key] = {
			'hits_at_1'   : round(sub_dict['hits_at_1'], n_round),
			'hits_at_3'   : round(sub_dict['hits_at_3'], n_round),
			'hits_at_5'   : round(sub_dict['hits_at_5'], n_round),
			'hits_at_10'  : round(sub_dict['hits_at_10'], n_round),
			'mr'          : round(sub_dict['arithmetic_mean_rank'], n_round),
			'mrr'         : round(sub_dict['inverse_harmonic_mean_rank'], n_round),
			'adjusted_mr' : round(sub_dict['adjusted_arithmetic_mean_rank'], n_round),
			'adjusted_mrr': round(sub_dict['adjusted_inverse_harmonic_mean_rank'], n_round)}

	# Check if the JSON file exists
	if os.path.exists(metrics_file):
		existing_results = read_json(metrics_file)
		existing_results.update({"link prediction": results_eval})
		save_json(existing_results, metrics_file)
	else:
		save_json({"link prediction": results_eval}, metrics_file)

	logger.info(f"Evaluating model {model_name} complete")


def triple_classification(result: PipelineResult,
						  model_name: str,
						  noisy_triples_file: str,
						  metrics_file: str,
						  noise_ratio: float):
	# load model
	model = result.model

	logger.info(f"evaluating {model_name} trained with {noise_ratio} noise ratio dataset on triple classification")

	# load dataset gold
	train_original, val_original, test_original = get_train_val_test_from_dir(noisy_triples_file, 0, False)
	# load dataset random
	train_noisy, val_noisy, test_noisy = get_train_val_test_from_dir(noisy_triples_file, 1, False)

	train_factory, val_factory, test_factory = get_train_val_test_factory(train_original,
																		  val_original,
																		  test_original,
																		  False)

	train_factory_noisy, val_factory_noisy, test_factory_noisy = get_train_val_test_factory(train_noisy,
																							val_noisy,
																							test_noisy,
																							False)

	### INFERENCE ON ORIGINAL TESTING
	real_train_scores = get_scores(model, train_factory, result.training)
	real_train_center = get_center(real_train_scores)

	#### INFERENCE ON VALIDATION
	real_val_scores = get_scores(model, val_factory, result.training)
	real_val_center = get_center(real_val_scores)

	#### INFERENCE ON TESTING
	real_test_scores = get_scores(model, test_factory, result.training)
	real_test_center = get_center(real_test_scores)

	fake_val_scores = get_scores(model, val_factory_noisy, result.training)
	fake_val_center = get_center(fake_val_scores)

	fake_test_scores = get_scores(model, test_factory_noisy, result.training)
	fake_test_center = get_center(fake_test_scores)

	threshold = fake_val_center + ((real_val_center - fake_val_center) / 2)
	logger.info(f"classification threshold: {threshold}")

	y_true = [1 for _ in real_test_scores] + [0 for _ in fake_test_scores]
	y_pred = [1 if y >= threshold else 0 for y in real_test_scores] + [1 if y >= threshold else 0 for y in
																	   fake_test_scores]

	n_round = 4
	accuracy = round(metrics.accuracy_score(y_true=y_true, y_pred=y_pred), n_round)
	f1_macro = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)
	f1_pos = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=1), n_round)
	f1_neg = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=0), n_round)
	precision = round(metrics.precision_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)
	recall = round(metrics.recall_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)

	maximum = np.max(real_train_scores)
	minimum = np.min(fake_test_scores)
	if real_test_center > fake_test_center:
		norm_distance = abs(real_test_center - fake_test_center) / abs(maximum - minimum)
		norm_distance = round(norm_distance, n_round)
	else:
		norm_distance = float('inf')
		logger.warning("WARNING: real_testing_scores_center <= fake_testing_scores_center")

	# Compute Z-test (http://homework.uoregon.edu/pub/class/es202/ztest.html)
	# Z = (mean_1 - mean_2) / sqrt{ (std1/sqrt(N1))**2 + (std2/sqrt(N2))**2 }
	real_scores_error = (real_test_scores.std() / (np.sqrt(real_test_scores.shape[0]))) ** 2
	fake_scores_error = (fake_test_scores.std() / (np.sqrt(fake_test_scores.shape[0]))) ** 2

	Z_statistic = round((real_test_scores.mean() - fake_test_scores.mean()) / (
		np.sqrt(real_scores_error + fake_scores_error)), 2)

	results_eval = {
		"accuracy"     : accuracy,
		"f1_macro"     : f1_macro,
		"f1_pos"       : f1_pos,
		"f1_neg"       : f1_neg,
		"precision"    : precision,
		"recall"       : recall,
		"Z_statistic"  : Z_statistic,
		"norm_distance": norm_distance}

	logger.info(results_eval)

	# Check if the JSON file exists
	if os.path.exists(metrics_file):
		existing_results = read_json(metrics_file)
		existing_results.update({"triple classification": results_eval})
		save_json(existing_results, metrics_file)
	else:
		save_json({"triple classification": results_eval}, metrics_file)

	logger.info(f"Evaluating model {model_name} complete")


# ======================== BERT EVALUATION  ======================== #
def link_deletion_bert(model: BertForSequenceClassification,
					   model_dir: str,
					   model_name: str,
					   noisy_triples_file: str,
					   metrics_file: str,
					   noise_ratio: float):
	logger.info(f"evaluating {model_name} trained with {noise_ratio} noise ratio dataset on link deletion")

	# load dataset gold
	train_original, val_original, test_original = get_train_val_test_from_dir(noisy_triples_file, 0)
	dataset_original = pd.concat([train_original, val_original, test_original], axis=0).reset_index(drop=True)
	test_original = pd.concat([val_original, test_original], axis=0).reset_index(drop=True)

	nodes = get_nodes(dataset_original)

	dataset_test = tokenize_and_generate_dataset(adjust_dataset_for_bert(test_original, label=int(True)))
	dataloader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=1)
	score_test = get_probabilities_bert(model, dataloader_test, device)

	ranks_head = []
	ranks_tail = []
	ranks = []

	# generate false connections
	for head, rel, tail in test_original.values.tolist():

		new_h = random.choice(nodes)
		# make sure node is not a real connection
		while [new_h, rel, tail] in dataset_original.values.tolist():
			new_h = random.choice(nodes)
		fake_head_triple = [new_h, rel, tail]

		new_t = random.choice(nodes)
		# make sure node is not a real connection
		while [head, rel, new_t] in dataset_original.values.tolist():
			new_t = random.choice(nodes)
		fake_tail_triple = [head, rel, new_t]

		fake_head_triples = pd.DataFrame([fake_head_triple], columns=['subject', 'predicate', 'object'])
		fake_tail_triples = pd.DataFrame([fake_tail_triple], columns=['subject', 'predicate', 'object'])

		# Load the BERT tokenizer
		dataset_fake_head = tokenize_and_generate_dataset(adjust_dataset_for_bert(fake_head_triples, label=int(False)))
		dataset_fake_tail = tokenize_and_generate_dataset(adjust_dataset_for_bert(fake_tail_triples, label=int(False)))

		dataloader_fake_head = DataLoader(dataset_fake_head, sampler=SequentialSampler(dataset_fake_head),
										  batch_size=1)
		dataloader_fake_tail = DataLoader(dataset_fake_tail, sampler=SequentialSampler(dataset_fake_tail),
										  batch_size=1)

		score_fake_head = get_probabilities_bert(model, dataloader_fake_head, device)
		score_fake_tail = get_probabilities_bert(model, dataloader_fake_tail, device)

		# scores are sorted in ascending order, meaning from the lowest to the highest
		# in link deletion we expect the score of the fake to be as low as possible (close to 0)
		# therefore close to the top of the list
		rank_head = np.searchsorted(a=score_test, v=score_fake_head, side='left') + 1
		rank_tail = np.searchsorted(a=score_test, v=score_fake_tail, side='left') + 1
		rank = int((rank_head + rank_tail) / 2.0)

		ranks_head.append(rank_head)
		ranks_tail.append(rank_tail)
		ranks.append(rank)

	# Metrics
	mr_calculator = ArithmeticMeanRank()
	adjusted_mr_calculator = AdjustedArithmeticMeanRank()
	mrr_calculator = InverseHarmonicMeanRank()
	adjusted_mrr_calculator = AdjustedInverseHarmonicMeanRank()
	hits_at_1_calculator = HitsAtK(k=1)
	hits_at_3_calculator = HitsAtK(k=3)
	hits_at_5_calculator = HitsAtK(k=5)
	hits_at_10_calculator = HitsAtK(k=10)

	test_size = len(test_original.values.tolist())
	n_round = 4

	ranks_head_array = np.array(ranks_head, dtype=int)
	ranks_tail_array = np.array(ranks_tail, dtype=int)

	# MR
	mr_head = round(float(mr_calculator(ranks_head_array, test_size)), n_round)
	mr_tail = round(float(mr_calculator(ranks_tail_array, test_size)), n_round)
	mr = round(int((mr_head + mr_tail) / 2.0), n_round)

	# ADJUSTED MR
	adjusted_mr_head = round(float(adjusted_mr_calculator(ranks_head_array, test_size)), n_round)
	adjusted_mr_tail = round(float(adjusted_mr_calculator(ranks_tail_array, test_size)), n_round)
	adjusted_mr = round(int((mr_head + mr_tail) / 2.0), n_round)

	# MRR
	mrr_head = round(float(mrr_calculator(ranks_head_array, test_size)), n_round)
	mrr_tail = round(float(mrr_calculator(ranks_tail_array, test_size)), n_round)
	mrr = round(float((mrr_head + mrr_tail) / 2.0), n_round)

	# ADJUSTED MRR
	adjusted_mrr_head = round(float(adjusted_mrr_calculator(ranks_head_array, test_size)), n_round)
	adjusted_mrr_tail = round(float(adjusted_mrr_calculator(ranks_tail_array, test_size)), n_round)
	adjusted_mrr = round(float((mrr_head + mrr_tail) / 2.0), n_round)

	# HITS AT 1
	hits_at_1_head = round(float(hits_at_1_calculator(ranks_head_array)), n_round)
	hits_at_1_tail = round(float(hits_at_1_calculator(ranks_tail_array)), n_round)
	hits_at_1 = round(float((hits_at_1_head + hits_at_1_tail) / 2.0), n_round)

	# HITS AT 3
	hits_at_3_head = round(float(hits_at_3_calculator(ranks_head_array)), n_round)
	hits_at_3_tail = round(float(hits_at_3_calculator(ranks_tail_array)), n_round)
	hits_at_3 = round(float((hits_at_3_head + hits_at_3_tail) / 2.0), n_round)

	# HITS AT 5
	hits_at_5_head = round(float(hits_at_5_calculator(ranks_head_array)), n_round)
	hits_at_5_tail = round(float(hits_at_5_calculator(ranks_tail_array)), n_round)
	hits_at_5 = round(float((hits_at_5_head + hits_at_5_tail) / 2.0), n_round)

	# HITS AT 10
	hits_at_10_head = round(float(hits_at_10_calculator(ranks_head_array)), n_round)
	hits_at_10_tail = round(float(hits_at_10_calculator(ranks_tail_array)), n_round)
	hits_at_10 = round(float((hits_at_10_head + hits_at_10_tail) / 2.0), n_round)

	results_eval = {
		'head': {
			'hits_at_1'   : hits_at_1_head,
			'hits_at_3'   : hits_at_3_head,
			'hits_at_5'   : hits_at_5_head,
			'hits_at_10'  : hits_at_10_head,
			'mr'          : mr_head,
			'adjusted_mr' : adjusted_mr_head,
			'mrr'         : mrr_head,
			'adjusted_mrr': adjusted_mrr_head},
		'both': {
			'hits_at_1'   : hits_at_1,
			'hits_at_3'   : hits_at_3,
			'hits_at_5'   : hits_at_5,
			'hits_at_10'  : hits_at_10,
			'mr'          : mr,
			'adjusted_mr' : adjusted_mr,
			'mrr'         : mrr,
			'adjusted_mrr': adjusted_mrr},
		'tail': {
			'hits_at_1'   : hits_at_1_tail,
			'hits_at_3'   : hits_at_3_tail,
			'hits_at_5'   : hits_at_5_tail,
			'hits_at_10'  : hits_at_10_tail,
			'mr'          : mr_tail,
			'adjusted_mr' : adjusted_mr_tail,
			'mrr'         : mrr_tail,
			'adjusted_mrr': adjusted_mrr_tail},

	}

	# Check if the JSON file exists
	if os.path.exists(metrics_file):
		existing_results = read_json(metrics_file)
		existing_results.update({"link deletion": results_eval})
		save_json(existing_results, metrics_file)
	else:
		save_json({"link deletion": results_eval}, metrics_file)

	logger.info(f"Evaluating model {model_name} complete")


def link_prediction_bert(model: BertForSequenceClassification,
						 model_dir: str,
						 model_name: str,
						 noisy_triples_file: str,
						 metrics_file: str,
						 noise_ratio: float):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	logger.info(f"evaluating {model_name} trained with {noise_ratio} noise ratio dataset on link prediction")

	# load original dataset
	train, val, test = get_train_val_test_from_dir(noisy_triples_file, 0)
	test = pd.concat([val, test], axis=0).reset_index(drop=True)

	ranks_head = []
	ranks_tail = []
	ranks = []

	dataset_original = pd.concat([train, val, test], axis=0).reset_index(drop=True)
	nodes = get_nodes(dataset_original)

	# for each real triple, sample 15 negatives
	for head, rel, tail in test.values.tolist():
		fake_head_triples = []
		fake_tail_triples = []

		# get a triple [?, rel, tail] where ? corresponds to a fake head
		while len(fake_head_triples) < 15:
			new_head = random.choice(nodes)
			# make sure it's fake
			while [new_head, rel, tail] in dataset_original.values.tolist():
				new_head = random.choice(nodes)

			fake_head_triples.append([new_head, rel, tail])

		# get a triple [head, rel, ?] where ? corresponds to a fake tail
		while len(fake_tail_triples) < 15:
			new_tail = random.choice(nodes)

			# make sure it's fake
			while [head, rel, new_tail] in dataset_original.values.tolist():
				new_tail = random.choice(nodes)

			fake_tail_triples.append([head, rel, new_tail])

		fake_head_triples = pd.DataFrame(fake_head_triples, columns=['subject', 'predicate', 'object'])
		fake_tail_triples = pd.DataFrame(fake_tail_triples, columns=['subject', 'predicate', 'object'])
		real_triple = pd.DataFrame([[head, rel, tail]], columns=['subject', 'predicate', 'object'])

		# create datasets
		fake_head_triples = adjust_dataset_for_bert(fake_head_triples, label=int(False))
		fake_tail_triples = adjust_dataset_for_bert(fake_tail_triples, label=int(False))
		real_triple = adjust_dataset_for_bert(real_triple, label=int(True))

		# Load the BERT tokenizer
		dataset_fake_head = tokenize_and_generate_dataset(fake_head_triples)
		dataset_fake_tail = tokenize_and_generate_dataset(fake_tail_triples)
		dataset_real_triple = tokenize_and_generate_dataset(real_triple)

		dataloader_fake_head = DataLoader(dataset_fake_head, sampler=SequentialSampler(dataset_fake_head),
										  batch_size=1)
		dataloader_fake_tail = DataLoader(dataset_fake_tail, sampler=SequentialSampler(dataset_fake_tail),
										  batch_size=1)
		dataloader_real_triple = DataLoader(dataset_real_triple,
											sampler=SequentialSampler(dataset_real_triple),
											batch_size=1)

		score_fake_head = get_probabilities_bert(model, dataloader_fake_head, device)
		score_fake_tail = get_probabilities_bert(model, dataloader_fake_tail, device)
		score_real_triple = get_probabilities_bert(model, dataloader_real_triple, device)

		# scores are sorted in ascending order, meaning from the lowest to the highest
		# in link prediction we expect the score of the real to be as high as possible
		# therefore close to the bottom -> invert results to get hits@k metrics
		rank_head = len(score_fake_head) - np.searchsorted(a=score_fake_head[::-1],
														   v=score_real_triple,
														   side='right') + 1
		rank_tail = len(score_fake_tail) - np.searchsorted(a=score_fake_tail[::-1],
														   v=score_real_triple,
														   side='right') + 1

		rank = int((rank_head + rank_tail) / 2.0)
		ranks_head.append(rank_head)
		ranks_tail.append(rank_tail)
		ranks.append(rank)

	# Metrics
	mr_calculator = ArithmeticMeanRank()
	adjusted_mr_calculator = AdjustedArithmeticMeanRank()
	mrr_calculator = InverseHarmonicMeanRank()
	adjusted_mrr_calculator = AdjustedInverseHarmonicMeanRank()
	hits_at_1_calculator = HitsAtK(k=1)
	hits_at_3_calculator = HitsAtK(k=3)
	hits_at_5_calculator = HitsAtK(k=5)
	hits_at_10_calculator = HitsAtK(k=10)

	test_size = len(test.values.tolist())
	n_round = 4

	ranks_head_array = np.array(ranks_head, dtype=int)
	ranks_tail_array = np.array(ranks_tail, dtype=int)

	# MR
	mr_head = round(float(mr_calculator(ranks_head_array, test_size)), n_round)
	mr_tail = round(float(mr_calculator(ranks_tail_array, test_size)), n_round)
	mr = round(int((mr_head + mr_tail) / 2.0), n_round)

	# ADJUSTED MR
	adjusted_mr_head = round(float(adjusted_mr_calculator(ranks_head_array, test_size)), n_round)
	adjusted_mr_tail = round(float(adjusted_mr_calculator(ranks_tail_array, test_size)), n_round)
	adjusted_mr = round(int((mr_head + mr_tail) / 2.0), n_round)

	# MRR
	mrr_head = round(float(mrr_calculator(ranks_head_array, test_size)), n_round)
	mrr_tail = round(float(mrr_calculator(ranks_tail_array, test_size)), n_round)
	mrr = round(float((mrr_head + mrr_tail) / 2.0), n_round)

	# ADJUSTED MRR
	adjusted_mrr_head = round(float(adjusted_mrr_calculator(ranks_head_array, test_size)), n_round)
	adjusted_mrr_tail = round(float(adjusted_mrr_calculator(ranks_tail_array, test_size)), n_round)
	adjusted_mrr = round(float((mrr_head + mrr_tail) / 2.0), n_round)

	# HITS AT 1
	hits_at_1_head = round(float(hits_at_1_calculator(ranks_head_array)), n_round)
	hits_at_1_tail = round(float(hits_at_1_calculator(ranks_tail_array)), n_round)
	hits_at_1 = round(float((hits_at_1_head + hits_at_1_tail) / 2.0), n_round)

	# HITS AT 3
	hits_at_3_head = round(float(hits_at_3_calculator(ranks_head_array)), n_round)
	hits_at_3_tail = round(float(hits_at_3_calculator(ranks_tail_array)), n_round)
	hits_at_3 = round(float((hits_at_3_head + hits_at_3_tail) / 2.0), n_round)

	# HITS AT 5
	hits_at_5_head = round(float(hits_at_5_calculator(ranks_head_array)), n_round)
	hits_at_5_tail = round(float(hits_at_5_calculator(ranks_tail_array)), n_round)
	hits_at_5 = round(float((hits_at_5_head + hits_at_5_tail) / 2.0), n_round)

	# HITS AT 10
	hits_at_10_head = round(float(hits_at_10_calculator(ranks_head_array)), n_round)
	hits_at_10_tail = round(float(hits_at_10_calculator(ranks_tail_array)), n_round)
	hits_at_10 = round(float((hits_at_10_head + hits_at_10_tail) / 2.0), n_round)

	results_eval = {
		'head': {
			'hits_at_1'   : hits_at_1_head,
			'hits_at_3'   : hits_at_3_head,
			'hits_at_5'   : hits_at_5_head,
			'hits_at_10'  : hits_at_10_head,
			'mr'          : mr_head,
			'adjusted_mr' : adjusted_mr_head,
			'mrr'         : mrr_head,
			'adjusted_mrr': adjusted_mrr_head},
		'both': {
			'hits_at_1'   : hits_at_1,
			'hits_at_3'   : hits_at_3,
			'hits_at_5'   : hits_at_5,
			'hits_at_10'  : hits_at_10,
			'mr'          : mr,
			'adjusted_mr' : adjusted_mr,
			'mrr'         : mrr,
			'adjusted_mrr': adjusted_mrr},
		'tail': {
			'hits_at_1'   : hits_at_1_tail,
			'hits_at_3'   : hits_at_3_tail,
			'hits_at_5'   : hits_at_5_tail,
			'hits_at_10'  : hits_at_10_tail,
			'mr'          : mr_tail,
			'adjusted_mr' : adjusted_mr_tail,
			'mrr'         : mrr_tail,
			'adjusted_mrr': adjusted_mrr_tail},

	}

	# Check if the JSON file exists
	if os.path.exists(metrics_file):
		existing_results = read_json(metrics_file)
		existing_results.update({"link prediction": results_eval})
		save_json(existing_results, metrics_file)
	else:
		save_json({"link prediction": results_eval}, metrics_file)

	logger.info(f"Evaluating model {model_name} complete")


def triple_classification_bert(model: BertForSequenceClassification,
							   model_dir: str,
							   model_name: str,
							   noisy_triples_file: str,
							   metrics_file: str,
							   noise_ratio: float):
	# load dataset gold
	train_original, val_original, test_original = get_train_val_test_from_dir(noisy_triples_file, 0)

	# get prediction for train gold
	dataset_train = tokenize_and_generate_dataset(adjust_dataset_for_bert(train_original, label=int(True)))
	dataloader_train = DataLoader(dataset_train, sampler=SequentialSampler(dataset_train), batch_size=1)
	score_train = np.array(get_probabilities_bert(model, dataloader_train, device))

	# get prediction for test gold
	test = pd.concat([val_original, test_original], axis=0).reset_index(drop=True)
	dataset_test = tokenize_and_generate_dataset(adjust_dataset_for_bert(test, label=int(True)))
	dataloader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=1)
	score_test = np.array(get_probabilities_bert(model, dataloader_test, device))

	# load dataset random
	train_noisy, val_noisy, test_noisy = get_train_val_test_from_dir(noisy_triples_file, 1)

	# get prediction for test noisy
	dataset_test_noisy = tokenize_and_generate_dataset(adjust_dataset_for_bert(test_noisy, label=int(False)))
	dataloader_test_noisy = DataLoader(dataset_test_noisy, sampler=SequentialSampler(dataset_test_noisy), batch_size=1)
	score_test_noisy = np.array(get_probabilities_bert(model, dataloader_test_noisy, device))

	# get prediction for val noisy
	dataset_val_noisy = tokenize_and_generate_dataset(adjust_dataset_for_bert(val_noisy, label=int(False)))
	dataloader_val_noisy = DataLoader(dataset_val_noisy, sampler=SequentialSampler(dataset_val_noisy), batch_size=1)
	score_val_noisy = np.array(get_probabilities_bert(model, dataloader_val_noisy, device))

	score_val_noisy_mean = get_center(score_val_noisy)
	score_test_mean = get_center(score_test)
	score_test_noisy_mean = get_center(score_test_noisy)

	threshold = score_val_noisy_mean + ((score_test_mean - score_val_noisy_mean) / 2)
	logger.info(f"classification threshold: {threshold}")

	y_true = [1 for _ in score_test] + [0 for _ in score_test_noisy]
	y_pred = [1 if y >= threshold else 0 for y in score_test] + [1 if y >= threshold else 0 for y in score_test_noisy]

	n_round = 4
	accuracy = round(metrics.accuracy_score(y_true=y_true, y_pred=y_pred), n_round)
	f1_macro = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)
	f1_pos = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=1), n_round)
	f1_neg = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=0), n_round)
	precision = round(metrics.precision_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)
	recall = round(metrics.recall_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)

	maximum = np.max(score_train)
	minimum = np.min(score_test_noisy)
	if score_test_mean > score_test_noisy_mean:
		norm_distance = abs(score_test_mean - score_test_noisy_mean) / abs(maximum - minimum)
	else:
		norm_distance = float('inf')
		logger.warning("WARNING: real_testing_scores_center <= fake_testing_scores_center")

	# Compute Z-test (http://homework.uoregon.edu/pub/class/es202/ztest.html)
	# Z = (mean_1 - mean_2) / sqrt{ (std1/sqrt(N1))**2 + (std2/sqrt(N2))**2 }
	real_scores_error = (score_test.std() / (np.sqrt(score_test.shape[0]))) ** 2
	fake_scores_error = (score_test_noisy.std() / (np.sqrt(score_test_noisy.shape[0]))) ** 2

	Z_statistic = round((score_test_mean - score_test_noisy_mean) / (np.sqrt(real_scores_error +
																			 fake_scores_error)), 2)

	results_eval = {
		"accuracy"     : round(float(accuracy), n_round),
		"f1_macro"     : round(float(f1_macro), n_round),
		"f1_pos"       : round(float(f1_pos), n_round),
		"f1_neg"       : round(float(f1_neg), n_round),
		"precision"    : round(float(precision), n_round),
		"recall"       : round(float(recall), n_round),
		"Z_statistic"  : round(float(Z_statistic), n_round),
		"norm_distance": round(float(norm_distance), n_round)}

	logger.info(results_eval)

	# Check if the JSON file exists
	if os.path.exists(metrics_file):
		existing_results = read_json(metrics_file)
		existing_results.update({"triple classification": results_eval})
		save_json(existing_results, metrics_file)
	else:
		save_json({"triple classification": results_eval}, metrics_file)

	logger.info(f"Evaluating model {model_name} complete")
