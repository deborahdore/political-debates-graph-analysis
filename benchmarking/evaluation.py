import os

import numpy as np
import pandas as pd
from loguru import logger
from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import PipelineResult
from pykeen.predict import predict_triples
from pykeen.triples import TriplesFactory

from benchmarking.utils.evaluation_utils import load_and_prepare_for_triple_classification
from utils.metrics import hits_at_k, mean_rank, mean_reciprocal_rank
from utils.dataset_utils import get_factory, get_nodes
from utils.evaluation_utils import corrupt_heads, corrupt_tails
from utils.utils import load, read_json, save_json

from sklearn import metrics


def link_deletion_evaluation(result: PipelineResult,
							 model_name: str,
							 triples_file: str,
							 metrics_file: str,
							 ratio: float):
	# load pytorch model
	model = result.model

	logger.info(f"evaluating {model_name} of dataset with {ratio} noise on link deletion")

	original_df_test = load(triples_file)
	original_df_test = pd.DataFrame(original_df_test[1:], columns=original_df_test[0])

	hits_at_1_head = 0
	hits_at_3_head = 0
	hits_at_5_head = 0
	hits_at_10_head = 0
	mr_head = 0
	mrr_head = 0

	hits_at_1_tail = 0
	hits_at_3_tail = 0
	hits_at_5_tail = 0
	hits_at_10_tail = 0
	mr_tail = 0
	mrr_tail = 0

	nodes = get_nodes(original_df_test)

	for idx, (h, r, t) in original_df_test.iterrows():
		triple = pd.DataFrame([[h, r, t]], columns=original_df_test.columns)

		# for every triple (h, r, t) create a new triple (h', r, t) where h' correspond to every node in the dataset
		head_fake = corrupt_heads(triple, nodes)
		tail_fake = corrupt_tails(triple, nodes)

		# create triples factory
		head_fake_factory = get_factory(head_fake)
		tail_fake_factory = get_factory(tail_fake)

		# get scores
		head_scores = predict_triples(model=model, triples=head_fake_factory.mapped_triples, mode=None).process(
			head_fake_factory).df.sort_values(by='score', ascending=False)

		tail_scores = predict_triples(model=model, triples=tail_fake_factory.mapped_triples, mode=None).process(
			tail_fake_factory).df.sort_values(by='score', ascending=False)

		hits_at_1_head += hits_at_k(predicted_entities=head_scores, true_entities=original_df_test, k=1)
		hits_at_3_head += hits_at_k(predicted_entities=head_scores, true_entities=original_df_test, k=3)
		hits_at_5_head += hits_at_k(predicted_entities=head_scores, true_entities=original_df_test, k=5)
		hits_at_10_head += hits_at_k(predicted_entities=head_scores, true_entities=original_df_test, k=10)
		mr_head += mean_rank(predicted=head_scores, true_entities=original_df_test)
		mrr_head += mean_reciprocal_rank(predicted_entities=head_scores, true_entities=original_df_test)

		hits_at_1_tail += hits_at_k(predicted_entities=tail_scores, true_entities=original_df_test, k=1)
		hits_at_3_tail += hits_at_k(predicted_entities=tail_scores, true_entities=original_df_test, k=3)
		hits_at_5_tail += hits_at_k(predicted_entities=tail_scores, true_entities=original_df_test, k=5)
		hits_at_10_tail += hits_at_k(predicted_entities=tail_scores, true_entities=original_df_test, k=10)
		mr_tail += mean_rank(predicted=tail_scores, true_entities=original_df_test)
		mrr_tail += mean_reciprocal_rank(predicted_entities=tail_scores, true_entities=original_df_test)

		results_eval = {
			"head": {
				"hits_at_1" : hits_at_1_head / len(original_df_test),
				"hits_at_3" : hits_at_3_head / len(original_df_test),
				"hits_at_5" : hits_at_5_head / len(original_df_test),
				"hits_at_10": hits_at_10_head / len(original_df_test),
				"mr"        : mr_head / len(original_df_test),
				"mrr"       : mrr_head / len(original_df_test)},
			"tail": {
				"hits_at_1" : hits_at_1_tail / len(original_df_test),
				"hits_at_3" : hits_at_3_tail / len(original_df_test),
				"hits_at_5" : hits_at_5_tail / len(original_df_test),
				"hits_at_10": hits_at_10_tail / len(original_df_test),
				"mr"        : mr_tail / len(original_df_test),
				"mrr"       : mrr_tail / len(original_df_test)}}

		# Check if the JSON file exists
		if os.path.isfile(metrics_file):
			existing_results = read_json(metrics_file)
			existing_results.update({"link deletion": results_eval})
			save_json(existing_results, metrics_file)
		else:
			save_json({"link deletion": results_eval}, metrics_file)

		logger.info(f"Evaluating model {model_name} complete")


def link_prediction_evaluation(result: PipelineResult,
							   noisy_train: TriplesFactory,
							   noisy_val: TriplesFactory,
							   original_test_file: str,
							   model_name: str,
							   metrics_file: str,
							   ratio: float):
	logger.info(f"evaluating {model_name} of dataset with {ratio} noise on link prediction")

	original_test = load(original_test_file)
	original_test_df = pd.DataFrame(original_test[1:], columns=original_test[0])
	original_test_df_factory = get_factory(original_test_df)

	evaluator = RankBasedEvaluator(metrics=["hits_at_k", "mr", "mrr"],
								   metrics_kwargs=[{'k': k} if metric == "hits_at_k" else {} for metric, k in
												   zip(["hits_at_k", "mr", "mrr"], (1, 3, 5, 10))])

	results_eval = evaluator.evaluate(model=result.model,
									  mapped_triples=original_test_df_factory.mapped_triples,
									  additional_filter_triples=[noisy_train.mapped_triples, noisy_val.mapped_triples],
									  batch_size=result.configuration.get('batch_size'),
									  use_tqdm=True,
									  slice_size=None).to_dict()

	# Check if the JSON file exists
	if os.path.isfile(metrics_file):
		existing_results = read_json(metrics_file)
		existing_results.update({"link prediction": results_eval})
		save_json(existing_results, metrics_file)
	else:
		save_json({"link prediction": results_eval}, metrics_file)

	logger.info(f"Evaluating model {model_name} complete")


def triple_classification(result: PipelineResult, model_name: str, triples_file: str, metrics_file: str):
	# load model
	model = result.model

	# generate dataset TRAINING
	real_train, real_train_factory = load_and_prepare_for_triple_classification(triples_file, "train")

	# generate datasets VALIDATION
	real_val, real_val_factory = load_and_prepare_for_triple_classification(triples_file, "val")
	fake_val, fake_val_factory = load_and_prepare_for_triple_classification(triples_file, "val", noisy=True)

	# generate dataset TESTING
	real_test, real_test_factory = load_and_prepare_for_triple_classification(triples_file, "test")
	fake_test, fake_test_factory = load_and_prepare_for_triple_classification(triples_file, "test", noisy=True)

	### INFERENCE ON ORIGINAL TESTING
	real_train_scores = predict_triples(model=model, triples=real_train_factory.mapped_triples, mode=None).process(
		real_train_factory).df.sort_values(by='score', ascending=False)
	real_train_center = float(np.mean(real_train_scores['score']))

	#### INFERENCE ON VALIDATION
	real_val_scores = predict_triples(model=model, triples=real_val_factory.mapped_triples, mode=None).process(
		real_val_factory).df.sort_values(by='score', ascending=False)
	real_val_center = float(np.mean(real_val_scores['score']))

	fake_val_scores = predict_triples(model=model, triples_factory=fake_val_factory.mapped_triples, mode=None).process(
		fake_val_factory).df.sort_values(by='score', ascending=False)
	fake_val_center = float(np.mean(fake_val_scores['score']))

	#### INFERENCE ON TESTING
	real_test_scores = predict_triples(model=model, triples=real_test_factory.mapped_triples, mode=None).process(
		real_test_factory).df.sort_values(by='score', ascending=False)
	real_test_center = float(np.mean(real_test_factory['score']))

	fake_test_scores = predict_triples(model=model,
									   triples_factory=fake_test_factory.mapped_triples,
									   mode=None).process(fake_test_factory).df.sort_values(by='score',
																							ascending=False)
	fake_test_center = float(np.mean(fake_test_factory['score']))

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
	if os.path.isfile(metrics_file):
		existing_results = read_json(metrics_file)
		existing_results.update({"triple classification": results_eval})
		save_json(existing_results, metrics_file)
	else:
		save_json({"triple classification": results_eval}, metrics_file)

	logger.info(f"Evaluating model {model_name} complete")
