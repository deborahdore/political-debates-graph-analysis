import os.path

import pandas as pd
from loguru import logger
from pykeen.predict import predict_triples

from utils.dataset_utils import get_factory, get_nodes
from utils.evaluation_utils import exchange_head, exchange_tail
from utils.metrics import hits_at_k_head, mean_rank_head, mean_reciprocal_rank
from utils.utils import load, load_model


def link_prediction_evaluation(model_name: str, model_dir: str, triples_file: str, ratio: float):
	# load pytorch model
	model = load_model(os.path.join(model_dir, f"{model_name}_{ratio}.pt"))

	original_df_test = load(triples_file.format(use="test"))
	original_df_test = pd.DataFrame(original_df_test[1:], columns=original_df_test[0])

	original_df_all = load(triples_file.format(use="all"))
	original_df_all = pd.DataFrame(original_df_all[1:], columns=original_df_all[0])

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

	nodes = get_nodes(original_df_all)

	for idx, (h, r, t) in original_df_test.iterrows():
		triple = pd.DataFrame([[h, r, t]], columns=original_df_test.columns)

		# for every triple (h, r, t) create a new triple (h', r, t) where h' correspond to every node in the dataset
		head_fake = exchange_head(triple, nodes)
		tail_fake = exchange_tail(triple, nodes)

		# create triples factory
		head_fake_factory = get_factory(head_fake)
		tail_fake_factory = get_factory(tail_fake)

		# get scores
		head_scores = predict_triples(model=model, triples=head_fake_factory.mapped_triples, mode=None).process(
			head_fake_factory).df.sort_values(by='score', ascending=False)

		tail_scores = predict_triples(model=model, triples=tail_fake_factory.mapped_triples, mode=None).process(
			tail_fake_factory).df.sort_values(by='score', ascending=False)

		hits_at_1_head += hits_at_k_head(true_entities=triple, predicted_entities=head_scores, k=1)
		hits_at_3_head += hits_at_k_head(true_entities=triple, predicted_entities=head_scores, k=3)
		hits_at_5_head += hits_at_k_head(true_entities=triple, predicted_entities=head_scores, k=5)
		hits_at_10_head += hits_at_k_head(true_entities=triple, predicted_entities=head_scores, k=10)
		mr_head += mean_rank_head(true_entities=triple, predicted_entities=head_scores)
		mrr_head += mean_reciprocal_rank(true_entities=triple, predicted_entities=head_scores)

		hits_at_1_tail += hits_at_k_head(true_entities=triple, predicted_entities=tail_scores, k=1)
		hits_at_3_tail += hits_at_k_head(true_entities=triple, predicted_entities=tail_scores, k=3)
		hits_at_5_tail += hits_at_k_head(true_entities=triple, predicted_entities=tail_scores, k=5)
		hits_at_10_tail += hits_at_k_head(true_entities=triple, predicted_entities=tail_scores, k=10)
		mr_tail += mean_rank_head(true_entities=triple, predicted_entities=tail_scores)
		mrr_tail += mean_reciprocal_rank(true_entities=triple, predicted_entities=tail_scores)

	hits_at_1_head = hits_at_1_head / len(original_df_test)
	hits_at_3_head = hits_at_3_head / len(original_df_test)
	hits_at_5_head = hits_at_5_head / len(original_df_test)
	hits_at_10_head = hits_at_10_head / len(original_df_test)
	mr_head = mr_head / len(original_df_test)
	mrr_head = mrr_head / len(original_df_test)

	hits_at_1_tail = hits_at_1_tail / len(original_df_test)
	hits_at_3_tail = hits_at_3_tail / len(original_df_test)
	hits_at_5_tail = hits_at_5_tail / len(original_df_test)
	hits_at_10_tail = hits_at_10_tail / len(original_df_test)
	mr_tail = mr_tail / len(original_df_test)
	mrr_tail = mrr_tail / len(original_df_test)

	# TODO to save

	logger.info("evaluation completed")


def link_deletion_evaluation(model_name: str, model_dir: str, triples_file: str, ratio: float):
	pass


def triple_evaluation(model_name: str, model_dir: str, triples_file: str, ratio: float):
	pass
