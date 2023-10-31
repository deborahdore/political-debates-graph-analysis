import os
import random
from typing import Any

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
from sklearn import metrics
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertForSequenceClassification

from utils.dataset_utils import get_nodes, get_train_val_test_factory, get_train_val_test_from_dir
from utils.evaluation_utils import adjust_dataset_for_bert, \
	get_center, \
	get_probabilities_bert, \
	get_scores, \
	get_scores_tensor, \
	tokenize_and_generate_dataset
from utils.utils import read_json, save_json

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# ======================== KGE MODELS EVALUATION  ======================== #
def link_deletion(model: Any,
				  model_name: str,
				  noisy_triples_file: str,
				  triplets_file_utils: str,
				  metrics_file: str,
				  noise_ratio: float):
	"""
	The link_deletion function is used to evaluate the performance of a model on link deletion.

	:param model: Any: Pass the model to the function
	:param model_name: str: Save the metrics in a json file
	:param noisy_triples_file: str: Specify the location of the noisy triples file
	:param triplets_file_utils: str: Location of entity-to-id and relation-to-id mappings
	:param metrics_file: str: Save the results of the link deletion experiment
	:param noise_ratio: float: Specify the amount of noise to be added to the dataset
	"""
	logger.info(f"## ====={model_name} trained with {noise_ratio} noise on link deletion ===== ##".upper())

	entity_to_id = read_json(triplets_file_utils.format(file_name="entity_to_id"))
	relation_to_id = read_json(triplets_file_utils.format(file_name="relation_to_id"))

	# load original dataset
	train_original, val_original, test_original = get_train_val_test_from_dir(noisy_triples_file,
																			  noise=0,
																			  drop_col_noise=True,
																			  get_noisy_test=False)
	# triples factories from original dataset
	train_factory, val_factory, test_factory = get_train_val_test_factory(train_original,
																		  val_original,
																		  test_original,
																		  triplets_file_utils,
																		  create_inverse_triples=False)

	real_test_scores = get_scores(model, test_factory)

	ranks = []
	ranks_head = []
	ranks_tail = []

	original_df = pd.concat([train_original, test_original, val_original], axis=0).reset_index(drop=True)
	nodes = get_nodes(original_df)

	for h, r, t in test_original.values.tolist():
		fake_head_triple = None
		fake_tail_triple = None

		# create fake head triple
		while True:
			new_h = random.choice(nodes)
			fake_head_triple = [new_h, r, t]
			if fake_head_triple not in original_df.values.tolist():
				break

		# create fake tail triple
		while True:
			new_t = random.choice(nodes)
			fake_tail_triple = [h, r, new_t]
			if fake_tail_triple not in original_df.values.tolist():
				break

		# get scores
		fake_score = get_scores_tensor(model, [fake_head_triple, fake_tail_triple], entity_to_id, relation_to_id)

		fake_h_score, fake_t_score = fake_score[0], fake_score[1]
		rank_head = np.searchsorted(a=real_test_scores, v=fake_h_score, side='left') + 1
		rank_tail = np.searchsorted(a=real_test_scores, v=fake_t_score, side='left') + 1
		ranks.append(int((rank_head + rank_tail) / 2.0))
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

	test_size = len(test_original)
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

	logger.info(f"## ===== LINK DELETION COMPLETE ===== ##")


def link_prediction(model: Any,
					noisy_triples_file: str,
					triplets_file_utils: str,
					model_name: str,
					metrics_file: str,
					noise_ratio: float):
	"""
	The link_prediction function evaluates the model on link prediction.

	:param model: torch.nn.Module: Get the model
	:param noisy_triples_file: str: Load the noisy triples file
	:param triplets_file_utils: str: Load the entity-to-id and relation-to-id mapping
	:param model_name: str: name of the model
	:param metrics_file: str: Save the results of the evaluation in a json file
	:param noise_ratio: float: Specify the noise ratio of the dataset

	"""
	logger.info(f"## ====={model_name} trained with {noise_ratio} noise on link prediction ===== ##".upper())

	# load original dataset
	train_original, val_original, test_original = get_train_val_test_from_dir(noisy_triples_file,
																			  noise=0,
																			  drop_col_noise=False,
																			  get_noisy_test=False)
	train_factory, val_factory, test_factory = get_train_val_test_factory(train_original,
																		  val_original,
																		  test_original,
																		  triplets_file_utils,
																		  create_inverse_triples=False)

	# load noisy dataset
	train_noisy, val_noisy, test_noisy = get_train_val_test_from_dir(noisy_triples_file,
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
									 device=device,
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

	# Check if the JSON file exists
	if os.path.exists(metrics_file):
		existing_results = read_json(metrics_file)
		existing_results.update({"link prediction": results_eval})
		save_json(existing_results, metrics_file)
	else:
		save_json({"link prediction": results_eval}, metrics_file)

	logger.info(f"## ===== LINK PREDICTION COMPLETE ===== ##")


def triple_classification(model: Any,
						  model_name: str,
						  noisy_triples_file: str,
						  triplets_file_utils: str,
						  metrics_file: str,
						  noise_ratio: float):
	"""
	The triple_classification function is used to evaluate the performance of a KGE model on triple classification.

	:param model: Any: Pass the model to be evaluated
	:param model_name: str: Identify the model in the metrics file
	:param noisy_triples_file: str: Specify the path to the directory containing all of the noisy triples
	:param triplets_file_utils: str: Location of entity-to-id and relation-to-id mappings
	:param metrics_file: str: Save the results of the triple classification
	:param noise_ratio: float: Specify the percentage of noise in the training set
	"""
	logger.info(f"## ====={model_name} trained with {noise_ratio} noise on triple classification ===== ##".upper())

	# ===== LOAD ORIGINAL
	train_original, val_original, test_original = get_train_val_test_from_dir(noisy_triples_file, 0, False)
	train_factory, val_factory, test_factory = get_train_val_test_factory(train_original,
																		  val_original,
																		  test_original,
																		  triplets_file_utils,
																		  create_inverse_triples=False)

	# ===== LOAD ALL FAKE
	train_fake, val_fake, test_fake = get_train_val_test_from_dir(noisy_triples_file,
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

	logger.info("Triples Classification statistics:".upper())
	logger.info(f"training_scores_center: {training_scores_center}")
	logger.info(f"fake_validation_scores_center: {fake_validation_scores_center}")
	logger.info(f"real_validation_scores_center: {real_validation_scores_center}")
	logger.info(f"fake_testing_scores_center: {fake_testing_scores_center}")
	logger.info(f"real_testing_scores_center: {real_testing_scores_center}")

	threshold = fake_validation_scores_center + ((real_validation_scores_center - fake_validation_scores_center) / 2)
	logger.info(f"classification threshold: {threshold}")

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
	f1_pos = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=1), n_round)
	logger.info(f"f1_pos: {f1_pos}")
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

	logger.info(f"## ===== TRIPLE CLASSIFICATION COMPLETE ===== ##")


# ======================== BERT EVALUATION  ======================== #
def link_deletion_bert(model: BertForSequenceClassification,
					   model_dir: str,
					   model_name: str,
					   noisy_triples_file: str,
					   metrics_file: str,
					   noise_ratio: float):
	logger.info(f"## ===== {model_name} trained with {noise_ratio} noise on link deletion ===== ##".upper())

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

	logger.info(f"## ===== LINK DELETION COMPLETE ===== ##".upper())


def link_prediction_bert(model: BertForSequenceClassification,
						 model_dir: str,
						 model_name: str,
						 noisy_triples_file: str,
						 metrics_file: str,
						 noise_ratio: float):
	logger.info(f"## ====={model_name} trained with {noise_ratio} noise on link predicion ===== ##".upper())

	# load original dataset
	train, val, test = get_train_val_test_from_dir(noisy_triples_file, 0)
	test = pd.concat([val, test], axis=0).reset_index(drop=True)

	train_noisy, val_noisy, test_noisy = get_train_val_test_from_dir(noisy_triples_file, 1)
	test_noisy = pd.concat([val_noisy, test_noisy], axis=0).reset_index(drop=True)
	fake_triples = pd.DataFrame(test_noisy, columns=['subject', 'predicate', 'object'])
	dataset_fake_triple = tokenize_and_generate_dataset(adjust_dataset_for_bert(fake_triples, label=int(False)))
	dataloader_fake_triple = DataLoader(dataset_fake_triple,
										sampler=SequentialSampler(dataset_fake_triple),
										batch_size=1)

	score_fake_triple = get_probabilities_bert(model, dataloader_fake_triple, device)

	ranks = []

	for head, rel, tail in test.values.tolist():
		real_triple = pd.DataFrame([[head, rel, tail]], columns=['subject', 'predicate', 'object'])

		# create datasets
		real_triple = adjust_dataset_for_bert(real_triple, label=int(True))

		# Load the BERT tokenizer
		dataset_real_triple = tokenize_and_generate_dataset(real_triple)

		dataloader_real_triple = DataLoader(dataset_real_triple,
											sampler=SequentialSampler(dataset_real_triple),
											batch_size=1)

		score_real_triple = get_probabilities_bert(model, dataloader_real_triple, device)

		# scores are sorted in ascending order, meaning from the lowest to the highest
		# in link prediction we expect the score of the real to be as high as possible
		# therefore close to the bottom -> invert results to get hits@k metrics
		rank = len(score_fake_triple) - np.searchsorted(a=score_fake_triple[::-1],
														v=score_real_triple,
														side='right') + 1

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
	n_round = 10

	ranks_array = np.array(ranks, dtype=int)

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

	# HITS AT 3
	hits_at_3 = round(float(hits_at_3_calculator(ranks_array)), n_round)

	# HITS AT 5
	hits_at_5 = round(float(hits_at_5_calculator(ranks_array)), n_round)

	# HITS AT 10
	hits_at_10 = round(float(hits_at_10_calculator(ranks_array)), n_round)

	results_eval = {
		'both': {
			'hits_at_1'   : hits_at_1,
			'hits_at_3'   : hits_at_3,
			'hits_at_5'   : hits_at_5,
			'hits_at_10'  : hits_at_10,
			'mr'          : mr,
			'adjusted_mr' : adjusted_mr,
			'mrr'         : mrr,
			'adjusted_mrr': adjusted_mrr}}

	# Check if the JSON file exists
	if os.path.exists(metrics_file):
		existing_results = read_json(metrics_file)
		existing_results.update({"link prediction": results_eval})
		save_json(existing_results, metrics_file)
	else:
		save_json({"link prediction": results_eval}, metrics_file)

	logger.info(f"## ===== LINK PREDICTION COMPLETE ===== ##".upper())


def triple_classification_bert(model: BertForSequenceClassification,
							   model_dir: str,
							   model_name: str,
							   noisy_triples_file: str,
							   metrics_file: str,
							   noise_ratio: float):
	logger.info(f"## ====={model_name} trained with {noise_ratio} noise on triple classification ===== ##".upper())

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
	train_noisy, val_noisy, test_noisy = get_train_val_test_from_dir(noisy_triples_file,
																	 100,
																	 drop_col_noise=False,
																	 get_noisy_test=True)
	train_noisy = train_noisy[train_noisy['noise'] == 1].copy()
	val_noisy = val_noisy[val_noisy['noise'] == 1].copy()
	test_noisy = test_noisy[test_noisy['noise'] == 1].copy()

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

	n_round = 10
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

	logger.info(f"## ===== EVALUATION COMPLETE ===== ##".upper())
