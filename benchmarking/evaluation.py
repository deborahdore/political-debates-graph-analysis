import os

import pandas as pd
from loguru import logger
from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import PipelineResult
from pykeen.predict import predict_triples
from pykeen.triples import TriplesFactory

from utils.metrics import hits_at_k, mean_rank, mean_reciprocal_rank
from utils.dataset_utils import generate_noise_for_triples_classification, get_factory, get_nodes
from utils.evaluation_utils import corrupt_heads, corrupt_tails
from utils.utils import load, read_json, save_json


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
		mr_head += mean_rank(predicted_entities=head_scores, true_entities=original_df_test)
		mrr_head += mean_reciprocal_rank(predicted_entities=head_scores, true_entities=original_df_test)

		hits_at_1_tail += hits_at_k(predicted_entities=tail_scores, true_entities=original_df_test, k=1)
		hits_at_3_tail += hits_at_k(predicted_entities=tail_scores, true_entities=original_df_test, k=3)
		hits_at_5_tail += hits_at_k(predicted_entities=tail_scores, true_entities=original_df_test, k=5)
		hits_at_10_tail += hits_at_k(predicted_entities=tail_scores, true_entities=original_df_test, k=10)
		mr_tail += mean_rank(predicted_entities=tail_scores, true_entities=original_df_test)
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


def triple_classification(result: PipelineResult, model_name: str, triples_file: str, metrics_file: str, ratio: float):
	# load model
	model = result.model

	original_test = load(triples_file)
	original_test_df = pd.DataFrame(original_test[1:], columns=original_test[0])

	# generate a dataset half good and half noisy
	noisy_test = generate_noise_for_triples_classification(triples_file)
	noisy_test_factory = get_factory(generate_noise_for_triples_classification(triples_file))

	predict_triples(model=model,
					triples=noisy_test_factory.mapped_triples,
					mode=None).process(noisy_test).df.sort_values(by='score', ascending=False)
