import os
import random

import numpy as np
import pandas as pd
from loguru import logger
from pykeen.evaluation import RankBasedEvaluator
from pykeen.metrics.ranking import AdjustedArithmeticMeanRank, \
	AdjustedInverseHarmonicMeanRank, \
	ArithmeticMeanRank, \
	HitsAtK, \
	InverseHarmonicMeanRank
from pykeen.pipeline import PipelineResult
from sklearn import metrics

from utils.dataset_utils import divide_dataset, \
	get_factory, \
	get_nodes, \
	get_train_val_test_factory, \
	get_train_val_test_from_dir
from utils.evaluation_utils import get_center, get_scores
from utils.utils import read_json, save_json


def link_deletion_evaluation(result: PipelineResult,
							 model_name: str,
							 noisy_triples_file: str,
							 metrics_file: str,
							 noise_ratio: float):
	# load pytorch model
	model = result.model

	# load dataset
	train_original, val_original, test_original = get_train_val_test_from_dir(noisy_triples_file, noise_ratio, False)

	# get real and noisy
	train, train_noisy = divide_dataset(train_original)
	val, val_noisy = divide_dataset(val_original)
	test, test_noisy = divide_dataset(test_original)

	# get factory
	train_factory, val_factory, test_factory = get_train_val_test_factory(train, val, test, False)

	logger.info(f"evaluating {model_name} of dataset with {noise_ratio} noise on link deletion")

	real_test_scores = get_scores(model, test_factory)

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

	nodes = get_nodes(val)

	for idx, (h, r, t) in test.iterrows():
		fake_head_triple = None
		fake_tail_triple = None

		while True:
			new_h = random.choice(nodes)
			while new_h == h:
				new_h = random.choice(nodes)
			fake_head_triple = [new_h, r, t]
			if fake_head_triple not in val.values.tolist():
				break
		while True:
			new_t = random.choice(nodes)
			fake_tail_triple = [h, r, new_t]
			if fake_tail_triple not in val.values.tolist():
				break

		fake_factory = get_factory(pd.DataFrame([fake_head_triple, fake_tail_triple], columns=train.columns))
		scores = get_scores(model, fake_factory)

		assert len(scores) == 2
		fake_h_score, fake_t_score = scores[0], scores[1]
		rank_head = np.searchsorted(a=real_test_scores.ravel(), v=fake_h_score.ravel(), side='left') + 1
		rank_tail = np.searchsorted(a=real_test_scores.ravel(), v=fake_t_score.ravel(), side='left') + 1
		ranks.append(int((rank_head + rank_tail) / 2.0))
		ranks_head.append(rank_head)
		ranks_tail.append(rank_tail)

	# Metrics
	test_size = len(test)
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
	logger.info(f"evaluating {model_name} of dataset with {noise_ratio} noise on link prediction")

	# load dataset
	train_original, val_original, test_original = get_train_val_test_from_dir(noisy_triples_file, noise_ratio, False)

	# get real and noisy
	train, train_noisy = divide_dataset(train_original)
	val, val_noisy = divide_dataset(val_original)
	test, test_noisy = divide_dataset(test_original)

	# get factory
	train_factory, val_factory, test_factory = get_train_val_test_factory(train, val, test, False)
	train_factory_noisy, val_factory_noisy, test_factory_noisy = get_train_val_test_factory(train_noisy,
																							val_noisy,
																							test_noisy,
																							False)

	evaluator = RankBasedEvaluator(metrics=["hits_at_k", "mr", "mrr"],
								   metrics_kwargs=[{'k': k} if metric == "hits_at_k" else {} for metric, k in
												   zip(["hits_at_k", "mr", "mrr"], (1, 3, 5, 10))])

	result_dict = evaluator.evaluate(model=result.model,
									 mapped_triples=test_factory.mapped_triples,
									 additional_filter_triples=[test_factory_noisy.mapped_triples,
																val_factory_noisy.mapped_triples],
									 batch_size=result.configuration.get('batch_size'),
									 use_tqdm=True,
									 slice_size=None).to_dict()

	results_eval = {}

	for key in ['head', 'both', 'tail']:
		sub_dict = result_dict[key]['realistic']
		results_eval[key] = {
			'hits_at_1'   : sub_dict['hits_at_1'],
			'hits_at_3'   : sub_dict['hits_at_3'],
			'hits_at_5'   : sub_dict['hits_at_5'],
			'hits_at_10'  : sub_dict['hits_at_10'],
			'mr'          : sub_dict['arithmetic_mean_rank'],
			'mrr'         : sub_dict['inverse_harmonic_mean_rank'],
			'adjusted_mr' : sub_dict['adjusted_arithmetic_mean_rank'],
			'adjusted_mrr': sub_dict['adjusted_inverse_harmonic_mean_rank']}

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
	# todo: what to do in this case
	# no fake/real nodes to test on
	if noise_ratio == 0 or noise_ratio == 1:
		return

	# load model
	model = result.model

	# load dataset
	train_original, val_original, test_original = get_train_val_test_from_dir(noisy_triples_file, noise_ratio, False)

	# get real and noisy
	train, train_noisy = divide_dataset(train_original)
	val, val_noisy = divide_dataset(val_original)
	test, test_noisy = divide_dataset(test_original)

	# get factory
	train_factory, val_factory, test_factory = get_train_val_test_factory(train, val, test, False)
	train_factory_noisy, val_factory_noisy, test_factory_noisy = get_train_val_test_factory(train_noisy,
																							val_noisy,
																							test_noisy,
																							False)
	### INFERENCE ON ORIGINAL TESTING
	real_train_scores = get_scores(model, train_factory)
	real_train_center = get_center(real_train_scores)

	#### INFERENCE ON VALIDATION
	real_val_scores = get_scores(model, val_factory)
	real_val_center = get_center(real_val_scores)

	fake_val_scores = get_scores(model, val_factory_noisy)
	fake_val_center = get_center(fake_val_scores)

	#### INFERENCE ON TESTING
	real_test_scores = get_scores(model, test_factory)
	real_test_center = get_center(real_test_scores)

	fake_test_scores = get_scores(model, test_factory_noisy)
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

	# compute norm_distance among the two distribution (greater is better)

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
