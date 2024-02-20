import os
import random

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from loguru import logger
from matplotlib import pyplot as plt
from pykeen.evaluation import RankBasedEvaluator
from pykeen.metrics.ranking import AdjustedArithmeticMeanRank, \
	AdjustedInverseHarmonicMeanRank, \
	ArithmeticMeanRank, \
	HitsAtK, \
	InverseHarmonicMeanRank
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from benchmarking import config
from benchmarking.utils.dataset_utils import get_nodes, get_train_val_test_factory, get_train_val_test_from_dir
from benchmarking.utils.evaluation_utils import get_center, get_scores, get_scores_tensor
from benchmarking.utils.util import read_json, read_tsv, save_json, save_tsv


# ======================== LINK DELETION  ======================== #

def link_deletion(model: torch.nn.Module,
				  model_name: str,
				  noisy_split_triplets_file: str,
				  triplets_file_utils: str,
				  task_name: str,
				  results_dir: str,
				  noise_ratio: float):
	"""	Evaluate the performance of a KGE model on link deletion."""

	logger.info(f"🎯 Evaluating {model_name} on link deletion")

	entity_to_id = read_json(triplets_file_utils.format(file_name="entity_to_id"))
	relation_to_id = read_json(triplets_file_utils.format(file_name="relation_to_id"))

	# load original dataset
	train_original, val_original, test_original = get_train_val_test_from_dir(noisy_split_triplets_file,
																			  noise=0,
																			  drop_col_noise=True,
																			  get_noisy_test=False)

	logger.info("📊 Relation counts: ")
	logger.info(test_original['relation'].value_counts())

	train_factory, val_factory, test_factory = get_train_val_test_factory(train_original,
																		  val_original,
																		  test_original,
																		  triplets_file_utils,
																		  create_inverse_triples=False)

	real_test_scores = get_scores(model, test_factory)

	ranks_head = []
	ranks_tail = []

	original_df = pd.concat([train_original, test_original, val_original],
							axis=0).dropna().drop_duplicates().reset_index(drop=True).values.tolist()

	if config.SPECIAL_BENCHMARKING_FLAG:
		relations = ['__label__Support', '__label__Attack', '__label__Equivalent']
	else:
		relations = train_original['relation'].drop_duplicates().values.tolist()

	nodes = get_nodes(train_original)
	test_size = len(test_original)

	for h, r, t in test_original.values.tolist():
		if r not in relations:
			test_size = test_size - 1
			continue

		# create fake head triple
		while True:
			new_h = random.sample(sorted(nodes), 1)[0]
			fake_head_triple = [new_h, r, t]
			if fake_head_triple not in original_df:
				break

		# create fake tail triple
		while True:
			new_t = random.sample(sorted(nodes), 1)[0]
			fake_tail_triple = [h, r, new_t]
			if fake_tail_triple not in original_df:
				break

		# get scores
		fake_score = get_scores_tensor(model=model,
									   triples=[fake_head_triple, fake_tail_triple],
									   entities_label_id_map=entity_to_id,
									   relation_label_id_map=relation_to_id)

		fake_h_score = fake_score[0]
		fake_t_score = fake_score[1]

		rank_head = np.searchsorted(a=real_test_scores, v=fake_h_score, side='left') + 1
		rank_tail = np.searchsorted(a=real_test_scores, v=fake_t_score, side='left') + 1

		ranks_head.append(rank_head)
		ranks_tail.append(rank_tail)

	# Metrics
	mr_calculator = ArithmeticMeanRank()
	adjusted_mr_calculator = AdjustedArithmeticMeanRank()
	mrr_calculator = InverseHarmonicMeanRank()
	adjusted_mrr_calculator = AdjustedInverseHarmonicMeanRank()
	hits_at_1_calculator = HitsAtK(k=1)
	hits_at_3_calculator = HitsAtK(k=3)
	hits_at_5_calculator = HitsAtK(k=5)
	hits_at_10_calculator = HitsAtK(k=10)

	n_round = 10

	ranks_head_array = np.array(ranks_head, dtype=int)
	ranks_tail_array = np.array(ranks_tail, dtype=int)

	# MR
	mr_head = round(float(mr_calculator(ranks_head_array, test_size)), n_round)
	mr_tail = round(float(mr_calculator(ranks_tail_array, test_size)), n_round)
	mr = round(int((mr_head + mr_tail) / 2.0), n_round)

	# ADJUSTED MR
	adjusted_mr_head = round(float(adjusted_mr_calculator(ranks_head_array, test_size)), n_round)
	adjusted_mr_tail = round(float(adjusted_mr_calculator(ranks_tail_array, test_size)), n_round)
	adjusted_mr = round(float((adjusted_mr_head + adjusted_mr_tail) / 2.0), n_round)

	# MRR
	mrr_head = round(float(mrr_calculator(ranks_head_array, test_size)), n_round)
	mrr_tail = round(float(mrr_calculator(ranks_tail_array, test_size)), n_round)
	mrr = round(float((mrr_head + mrr_tail) / 2.0), n_round)

	# ADJUSTED MRR
	adjusted_mrr_head = round(float(adjusted_mrr_calculator(ranks_head_array, test_size)), n_round)
	adjusted_mrr_tail = round(float(adjusted_mrr_calculator(ranks_tail_array, test_size)), n_round)
	adjusted_mrr = round(float((adjusted_mrr_head + adjusted_mrr_tail) / 2.0), n_round)

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
	logger.info(results_eval)

	result_dir = results_dir.format(model=model_name, task_name=task_name, noise=noise_ratio)
	save_json(results_eval, f"{result_dir}/link_deletion.json")

	logger.info(f"🎯 Link deletion complete 🎯")


# ======================== RELATION PREDICTION  ======================== #
def relation_prediction(model: torch.nn.Module,
						model_name: str,
						noisy_split_triplets_file: str,
						triplets_file_utils: str,
						task_name: str,
						results_dir: str,
						noise_ratio: float,
						relation_to_evaluate: str = None):
	""" Evaluate the performance of a KGE model on relation prediction """

	logger.info(f"🎯 Evaluating {model_name} on "
				f"{'relation' if relation_to_evaluate is None else relation_to_evaluate} "
				f"prediction")

	entity_to_id = read_json(triplets_file_utils.format(file_name="entity_to_id"))
	relation_to_id = read_json(triplets_file_utils.format(file_name="relation_to_id"))

	# load original dataset
	train_original, val_original, test_original = get_train_val_test_from_dir(noisy_split_triplets_file,
																			  noise=0,
																			  drop_col_noise=True,
																			  get_noisy_test=False)
	logger.info("📊 Relation counts: ")
	logger.info(test_original['relation'].value_counts())

	if config.SPECIAL_BENCHMARKING_FLAG:
		logger.info("🚀 Training with only relevant relations")
		relations = ['__label__Support', '__label__Attack', '__label__Equivalent']
	else:
		relations = train_original['relation'].drop_duplicates().values.tolist()

	test_size = len(test_original)
	ranks = []

	if relation_to_evaluate is not None:
		test_original = test_original[test_original['relation'] == relation_to_evaluate]

	for h, r, t in test_original.values.tolist():

		# create fake relation triple
		fake_triples = []
		for relation in relations:
			if relation == r: continue
			fake_triples.append([h, relation, t])

		# get scores
		real_score = get_scores_tensor(model=model,
									   triples=[[h, r, t]],
									   entities_label_id_map=entity_to_id,
									   relation_label_id_map=relation_to_id)

		fake_score = get_scores_tensor(model=model,
									   triples=fake_triples,
									   entities_label_id_map=entity_to_id,
									   relation_label_id_map=relation_to_id,
									   sort=True)

		rank = len(fake_score) - np.searchsorted(a=fake_score, v=real_score, side='right') + 1
		ranks.append(rank)

	# Metrics
	mr_calculator = ArithmeticMeanRank()
	adjusted_mr_calculator = AdjustedArithmeticMeanRank()
	mrr_calculator = InverseHarmonicMeanRank()
	adjusted_mrr_calculator = AdjustedInverseHarmonicMeanRank()
	hits_at_1_calculator = HitsAtK(k=1)
	hits_at_2_calculator = HitsAtK(k=2)

	n_round = 10
	ranks_array = np.array(ranks)

	# MR
	mr = round(float(mr_calculator(ranks_array, test_size)), n_round)

	# ADJUSTED MR
	adjusted_mr = round(float(adjusted_mr_calculator(ranks_array, test_size)), n_round)

	# MRR
	mrr = round(float(mrr_calculator(ranks_array, test_size)), n_round)

	# ADJUSTED MRR
	adjusted_mrr = round(float(adjusted_mrr_calculator(ranks_array, test_size)), n_round)

	# HITS AT 1
	hits_at_1 = round(float(hits_at_1_calculator(ranks_array)), n_round)

	# HITS AT 2
	hits_at_2 = round(float(hits_at_2_calculator(ranks_array)), n_round)

	results_eval = {
		'hits_at_1'   : hits_at_1,
		'hits_at_2'   : hits_at_2,
		'mr'          : mr,
		'mrr'         : mrr,
		'adjusted_mr' : adjusted_mr,
		'adjusted_mrr': adjusted_mrr}

	logger.info(results_eval)

	result_dir = results_dir.format(model=model_name, task_name=task_name, noise=noise_ratio)
	save_json(results_eval, f"{result_dir}/"
							f"{'relation' if relation_to_evaluate is None else relation_to_evaluate}_prediction.json")

	logger.info(f"🎯 {'Relation' if relation_to_evaluate is None else relation_to_evaluate} prediction complete 🎯")


# ======================== LINK PREDICTION  ======================== #
def link_prediction(model: torch.nn.Module,
					model_name: str,
					noisy_split_triplets_file: str,
					triplets_file_utils: str,
					task_name: str,
					results_dir: str,
					noise_ratio: float):
	""" Evaluates the performance of a KGE model on link prediction. """

	logger.info(f"🎯 Evaluating {model_name} on link prediction")

	# load original dataset
	train, val, test = get_train_val_test_from_dir(noisy_split_triplets_file,
												   noise=0,
												   drop_col_noise=False,
												   get_noisy_test=False)
	logger.info("📊 Relation counts: ")
	logger.info(test['relation'].value_counts())

	train_factory, val_factory, test_factory = get_train_val_test_factory(train,
																		  val,
																		  test,
																		  triplets_file_utils,
																		  create_inverse_triples=False)

	# load noisy dataset
	train_noisy, val_noisy, test_noisy = get_train_val_test_from_dir(noisy_split_triplets_file,
																	 noise=noise_ratio,
																	 drop_col_noise=False,
																	 get_noisy_test=True)
	train_factory_noisy, val_factory_noisy, test_factory_noisy = get_train_val_test_factory(train_noisy,
																							val_noisy,
																							test_noisy,
																							triplets_file_utils,
																							create_inverse_triples=False)
	# Launch evaluation pipeline
	evaluator = RankBasedEvaluator(filtered=True)
	result_dict = evaluator.evaluate(model=model,
									 mapped_triples=test_factory.mapped_triples,
									 additional_filter_triples=[train_factory_noisy.mapped_triples,
																# filter on training triples with noisy
																val_factory_noisy.mapped_triples,
																# filter on validation triples with noisy
																],

									 batch_size=None,
									 slice_size=None,
									 device=config.DEVICE,
									 use_tqdm=True).to_dict()

	results_eval = {}
	n_round = 10
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

	logger.info(results_eval)

	result_dir = results_dir.format(model=model_name, task_name=task_name, noise=noise_ratio)
	save_json(results_eval, f"{result_dir}/link_prediction.json")
	logger.info(f"🎯 Link prediction complete 🎯")


# ======================== TRIPLE CLASSIFICATION  ======================== #

def triple_classification(model: torch.nn.Module,
						  model_name: str,
						  noisy_split_triplets_file: str,
						  triplets_file_utils: str,
						  task_name: str,
						  results_dir: str,
						  noise_ratio: float):
	"""	Evaluate the performance of a KGE model on triple classification """
	logger.info(f"🎯 Evaluating {model_name} on triple classification")

	# ===== LOAD ORIGINAL
	train_original, val_original, test_original = get_train_val_test_from_dir(noisy_split_triplets_file, 0, False)

	logger.info("📊 Relation counts: ")
	logger.info(test_original['relation'].value_counts())

	train_factory, val_factory, test_factory = get_train_val_test_factory(train_original,
																		  val_original,
																		  test_original,
																		  triplets_file_utils,
																		  create_inverse_triples=False)

	# ===== LOAD ALL FAKE
	train_fake, val_fake, test_fake = get_train_val_test_from_dir(noisy_split_triplets_file,
																  100,
																  drop_col_noise=False,
																  get_noisy_test=True)
	train_fake = train_fake[train_fake['noise'] == str(1)].copy()
	val_fake = val_fake[val_fake['noise'] == str(1)].copy()
	test_fake = test_fake[test_fake['noise'] == str(1)].copy()
	train_fake_factory, val_fake_factory, test_fake_factory = get_train_val_test_factory(train_fake,
																						 val_fake,
																						 test_fake,
																						 triplets_file_utils,
																						 create_inverse_triples=False)

	# ===== Inference (computation of KGE scores) on Original Training Set ====== #
	# REAL
	training_scores_vector = get_scores(model=model, factory=train_factory)
	training_scores_center = get_center(scores=training_scores_vector)

	# ===== Inference (computation of KGE scores) on Validation Set ====== #
	# FAKE
	fake_validation_scores = get_scores(model=model, factory=val_fake_factory)
	fake_validation_scores_center = get_center(scores=fake_validation_scores)

	# REAL
	real_validation_scores = get_scores(model=model, factory=val_factory)
	real_validation_scores_center = get_center(scores=real_validation_scores)

	# ===== Inference (computation of KGE scores) on Testing Set ====== #
	# FAKE
	fake_testing_scores = get_scores(model=model, factory=test_fake_factory)
	fake_testing_scores_center = get_center(scores=fake_testing_scores)

	# REAL
	real_testing_scores = get_scores(model=model, factory=test_factory)
	real_testing_scores_center = get_center(scores=real_testing_scores)

	logger.info("🎯 Triples Classification statistics: 🎯")
	logger.info(f"📊 training_scores_center: {training_scores_center}")
	logger.info(f"📊 fake_validation_scores_center: {fake_validation_scores_center}")
	logger.info(f"📊 real_validation_scores_center: {real_validation_scores_center}")
	logger.info(f"📊 fake_testing_scores_center: {fake_testing_scores_center}")
	logger.info(f"📊 real_testing_scores_center: {real_testing_scores_center}")

	threshold = fake_validation_scores_center + ((real_validation_scores_center - fake_validation_scores_center) / 2)
	logger.info(f"📊 classification threshold: {threshold}")

	y_true = [1 for _ in real_testing_scores] + [0 for _ in fake_testing_scores]
	y_pred = [1 if y >= threshold else 0 for y in real_testing_scores] + [1 if y >= threshold else 0 for y in
																		  fake_testing_scores]
	assert len(y_pred) == len(y_true)
	assert sum(y_true) == len(real_testing_scores)
	assert sum(y_pred) <= len(y_true)
	assert sum(y_pred) >= 0

	n_round = 10
	accuracy = round(metrics.accuracy_score(y_true=y_true, y_pred=y_pred), n_round)
	logger.info(f"accuracy: {accuracy}")
	f1_macro = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)
	logger.info(f"f1_macro: {f1_macro}")
	f1_micro = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="micro"), n_round)
	logger.info(f"f1_micro: {f1_micro}")
	f1_pos = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=1), n_round)
	logger.info(f"f1_pos: {f1_pos}")
	f1_weighted = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="weighted"), n_round)
	logger.info(f"f1_weighted: {f1_weighted}")
	f1_neg = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=0), n_round)
	logger.info(f"f1_neg: {f1_neg}")
	precision = round(metrics.precision_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)
	logger.info(f"precision: {precision}")
	recall = round(metrics.recall_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)
	logger.info(f"recall: {recall}")

	# compute distance among the two distribution (greater is better)
	maximum = np.max(training_scores_vector)
	minimum = np.min(fake_testing_scores)
	if real_testing_scores_center > fake_testing_scores_center:
		norm_distance = round(abs(real_testing_scores_center - fake_testing_scores_center) / abs(maximum - minimum),
							  n_round)
		logger.info(f"distance: {norm_distance}")
	else:
		norm_distance = float('inf')
		logger.warning("WARNING: real_testing_scores_center <= fake_testing_scores_center")

	# Compute Z-test (http://homework.uoregon.edu/pub/class/es202/ztest.html)
	# Z = (mean_1 - mean_2) / sqrt{ (std1/sqrt(N1))**2 + (std2/sqrt(N2))**2 }
	real_scores_error = (real_testing_scores.std() / (np.sqrt(real_testing_scores.shape[0]))) ** 2
	fake_scores_error = (fake_testing_scores.std() / (np.sqrt(fake_testing_scores.shape[0]))) ** 2
	Z_statistic = round((
								real_testing_scores.mean() - fake_testing_scores.mean()) / np.sqrt(real_scores_error +
																								   fake_scores_error),
						2)
	logger.info(f"Z-statistic: {round(Z_statistic, n_round)}")

	results_eval = {
		"accuracy"     : accuracy,
		"f1_macro"     : f1_macro,
		"f1_micro"     : f1_micro,
		"f1_weighted"  : f1_weighted,
		"f1_pos"       : f1_pos,
		"f1_neg"       : f1_neg,
		"precision"    : precision,
		"recall"       : recall,
		"Z_statistic"  : Z_statistic,
		"norm_distance": norm_distance}

	logger.info(results_eval)

	results_dir = results_dir.format(model=model_name, task_name=task_name, noise=noise_ratio)
	save_json(results_eval, f"{results_dir}/triple_classification.json")

	logger.info("🎯 Triple classification complete 🎯")


# ======================== RELATION CLASSIFICATION  ======================== #

def relation_classification(model: torch.nn.Module,
							model_name: str,
							noisy_split_triplets_file: str,
							triplets_file_utils: str,
							task_name: str,
							results_dir: str,
							noise_ratio: float):
	"""	Evaluate the performance of a KGE model on relation classification.	"""

	logger.info(f"🎯 Evaluating {model_name} on relation classification")

	entity_to_id = read_json(triplets_file_utils.format(file_name="entity_to_id"))
	relation_to_id = read_json(triplets_file_utils.format(file_name="relation_to_id"))

	# ===== LOAD ORIGINAL
	train_original, val_original, test_original = get_train_val_test_from_dir(noisy_split_triplets_file, 0, True)

	logger.info("📊 Relation counts: ")
	logger.info(test_original['relation'].value_counts())

	train_factory, val_factory, test_factory = get_train_val_test_factory(train=train_original,
																		  val=val_original,
																		  test=test_original,
																		  triplets_file_utils=triplets_file_utils,
																		  create_inverse_triples=False)

	# ===== LOAD ALL FAKE
	train_fake, val_fake, test_fake = get_train_val_test_from_dir(noisy_split_triplets_file,
																  100,
																  drop_col_noise=False,
																  get_noisy_test=True)
	train_fake = train_fake[train_fake['noise'] == str(1)].copy()
	val_fake = val_fake[val_fake['noise'] == str(1)].copy()
	test_fake = test_fake[test_fake['noise'] == str(1)].copy()
	train_fake_factory, val_fake_factory, test_fake_factory = get_train_val_test_factory(train=train_fake,
																						 val=val_fake,
																						 test=test_fake,
																						 triplets_file_utils=triplets_file_utils,
																						 create_inverse_triples=False)

	# ===== Inference (computation of KGE scores) on Original Training Set ====== #
	# REAL
	training_scores_vector = get_scores(model=model, factory=train_factory)
	training_scores_center = get_center(scores=training_scores_vector)

	# ===== Inference (computation of KGE scores) on Validation Set ====== #
	# FAKE
	fake_validation_scores = get_scores(model=model, factory=val_fake_factory)
	fake_validation_scores_center = get_center(scores=fake_validation_scores)

	# REAL
	real_validation_scores = get_scores(model=model, factory=val_factory)
	real_validation_scores_center = get_center(scores=real_validation_scores)

	# ===== Inference (computation of KGE scores) on Testing Set ====== #
	# FAKE
	fake_testing_scores = get_scores(model=model, factory=test_fake_factory)
	fake_testing_scores_center = get_center(scores=fake_testing_scores)

	# REAL
	real_testing_scores = get_scores(model=model, factory=test_factory)
	real_testing_scores_center = get_center(scores=real_testing_scores)

	logger.info("🎯 Relation Classification statistics: 🎯")
	logger.info(f"📊training_scores_center: {training_scores_center}")
	logger.info(f"📊fake_validation_scores_center: {fake_validation_scores_center}")
	logger.info(f"📊real_validation_scores_center: {real_validation_scores_center}")
	logger.info(f"📊fake_testing_scores_center: {fake_testing_scores_center}")
	logger.info(f"📊real_testing_scores_center: {real_testing_scores_center}")

	threshold = fake_validation_scores_center + ((real_validation_scores_center - fake_validation_scores_center) / 2)
	logger.info(f"📊classification threshold: {threshold}")

	if config.SPECIAL_BENCHMARKING_FLAG:
		relations = ['__label__Support', '__label__Attack', '__label__Equivalent']
	else:
		relations = pd.concat([train_original, val_original, test_original], axis=0)[
			'relation'].drop_duplicates().values.tolist()

	y_true = []
	triples = []

	test_original = test_original.values.tolist()
	len_test = len(test_original)

	for head, rel, tail in test_original:
		if rel not in relations:
			len_test = len_test - 1
			continue

		for i, elem in enumerate(relations):
			triples.append([head, elem, tail])
			if elem == rel:
				y_true.append(1)
			else:
				y_true.append(0)

	scores = get_scores_tensor(model,
							   triples=triples,
							   entities_label_id_map=entity_to_id,
							   relation_label_id_map=relation_to_id)

	y_pred = [1 if y >= threshold else 0 for y in scores]

	assert len(y_pred) == len(y_true)

	n_round = 10
	accuracy = round(metrics.accuracy_score(y_true=y_true, y_pred=y_pred), n_round)
	logger.info(f"accuracy: {accuracy}")
	f1_macro = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)
	logger.info(f"f1_macro: {f1_macro}")
	f1_micro = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="micro"), n_round)
	logger.info(f"f1_micro: {f1_micro}")
	f1_pos = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=1), n_round)
	logger.info(f"f1_pos: {f1_pos}")
	f1_weighted = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="weighted"), n_round)
	logger.info(f"f1_weighted: {f1_weighted}")
	f1_neg = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=0), n_round)
	logger.info(f"f1_neg: {f1_neg}")
	precision = round(metrics.precision_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)
	logger.info(f"precision: {precision}")
	recall = round(metrics.recall_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)
	logger.info(f"recall: {recall}")

	# compute distance among the two distribution (greater is better)
	maximum = np.max(training_scores_vector)
	minimum = np.min(fake_testing_scores)
	if real_testing_scores_center > fake_testing_scores_center:
		norm_distance = round(abs(real_testing_scores_center - fake_testing_scores_center) / abs(maximum - minimum),
							  n_round)
		logger.info(f"distance: {norm_distance}")
	else:
		norm_distance = float('inf')
		logger.warning("WARNING: real_testing_scores_center <= fake_testing_scores_center")

	# Compute Z-test (http://homework.uoregon.edu/pub/class/es202/ztest.html)
	# Z = (mean_1 - mean_2) / sqrt{ (std1/sqrt(N1))**2 + (std2/sqrt(N2))**2 }
	real_scores_error = (real_testing_scores.std() / (np.sqrt(real_testing_scores.shape[0]))) ** 2
	fake_scores_error = (fake_testing_scores.std() / (np.sqrt(fake_testing_scores.shape[0]))) ** 2
	Z_statistic = round((
								real_testing_scores.mean() - fake_testing_scores.mean()) / np.sqrt(real_scores_error +
																								   fake_scores_error),
						2)
	logger.info(f"Z-statistic: {round(Z_statistic, n_round)}")

	results_eval = {
		"accuracy"     : accuracy,
		"f1_macro"     : f1_macro,
		"f1_micro"     : f1_micro,
		"f1_weighted"  : f1_weighted,
		"f1_pos"       : f1_pos,
		"f1_neg"       : f1_neg,
		"precision"    : precision,
		"recall"       : recall,
		"Z_statistic"  : Z_statistic,
		"norm_distance": norm_distance}

	logger.info(results_eval)

	results_dir = results_dir.format(model=model_name, task_name=task_name, noise=noise_ratio)
	save_json(results_eval, f"{results_dir}/relation_classification.json")

	logger.info("🎯 Relation classification complete 🎯")
	return threshold


# ======================== MAKE PREDICTIONS  ======================== #

def make_prediction(model: torch.nn.Module,
					model_name: str,
					noisy_split_triplets_file2: str,
					triplets_file_utils: str,
					task_name: str,
					results_dir: str,
					plot_dir: str,
					noise: int,
					threshold: float):
	""" Make and Save into csv the predictions on the test file """

	logger.info(f"🎯 Use {model_name} to make predictions")

	entity_to_id = read_json(triplets_file_utils.format(file_name="entity_to_id"))
	if config.MODE_TEXT == "text+type":
		new_entity_to_id = {}
		for idx, key in enumerate(entity_to_id):
			new_entity_to_id[key.split("_")[0]] = idx
		entity_to_id = new_entity_to_id

	relation_to_id = read_json(triplets_file_utils.format(file_name="relation_to_id"))
	test_original = read_tsv(noisy_split_triplets_file2.format(split="test"))

	y_true = test_original['relation'].map(lambda x: x.replace("__label__", "")).values.tolist()
	heads = test_original['head'].values.tolist()
	tails = test_original['tail'].values.tolist()
	y_pred = []

	relations = ['__label__Support', '__label__Attack', '__label__Equivalent']

	skipped = 0
	for head, rel, tail in test_original.values.tolist():
		triples = []
		for elem in relations:
			triples.append([head, elem, tail])

		try:
			scores = get_scores_tensor(model,
									   triples=triples,
									   entities_label_id_map=entity_to_id,
									   relation_label_id_map=relation_to_id)
		except Exception as err:
			logger.warning(f"{err} not preset between entities")
			y_pred.append("noRel")
			skipped += 1
			continue

		if max(scores) > threshold:
			# assign relation
			if scores.argmax() == 0:
				y_pred.append("Support")
			elif scores.argmax() == 1:
				y_pred.append("Attack")
			else:
				y_pred.append("noRel")
		else:
			y_pred.append("noRel")

	logger.info(f"Skipped: {skipped}")
	assert len(test_original) == len(y_true) == len(y_pred) == len(heads) == len(tails)

	predictions = pd.DataFrame({'true': y_true, 'predicted': y_pred, 'sentence1': heads, 'sentence2': tails})
	results_dir = results_dir.format(model=model_name, task_name=task_name, noise=noise)

	save_tsv(predictions, os.path.join(results_dir, "predictions.tsv"))

	n_round = 10
	accuracy = round(metrics.accuracy_score(y_true=y_true, y_pred=y_pred), n_round)
	logger.info(f"accuracy: {accuracy}")
	f1_macro = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)
	logger.info(f"f1_macro: {f1_macro}")
	f1_micro = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="micro"), n_round)
	logger.info(f"f1_micro: {f1_micro}")
	f1_weighted = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="weighted"), n_round)
	logger.info(f"f1_weighted: {f1_weighted}")
	precision = round(metrics.precision_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)
	logger.info(f"precision: {precision}")
	recall = round(metrics.recall_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)
	logger.info(f"recall: {recall}")

	value_dict = {"Support": 0, "Attack": 1, "noRel": 3}
	cf = confusion_matrix(predictions['true'].map(lambda x: value_dict.get(x)),
						  predictions['predicted'].map(lambda x: value_dict.get(x)))
	logger.info(f"confusion matrix: {cf}")

	results_eval = {
		"accuracy"   : accuracy,
		"f1_macro"   : f1_macro,
		"f1_micro"   : f1_micro,
		"f1_weighted": f1_weighted,
		"precision"  : precision,
		"recall"     : recall}

	save_json(results_eval, f"{results_dir}/predictions_eval.json")

	sns.heatmap(cf,
				annot=True,
				cmap='Blues',
				xticklabels=list(value_dict.keys()),
				yticklabels=list(value_dict.keys()),
				fmt='g')
	plt.xlabel('Predicted Labels')
	plt.ylabel('True Labels')
	plt.title(f'Confusion Matrix {model_name} - {noise}')

	plt.tight_layout()
	plot_dir = plot_dir.format(model=model_name, task_name=task_name, noise=noise)
	plt.savefig(os.path.join(plot_dir, "confusion_matrix.svg"))
