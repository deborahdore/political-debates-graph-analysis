import numpy as np
import pandas as pd
from pykeen.predict import predict_triples


def corrupt_heads(dataset: pd.DataFrame, nodes: pd.Series):
	final_triples = []
	for idx, (head, relation, tail) in dataset.iterrows():
		final_triples.append([head, relation, tail])  # append correct triple
		for new_head in nodes:
			final_triples.append([new_head, relation, tail])

	link_prediction_head_df = pd.DataFrame(final_triples, columns=dataset.columns)
	link_prediction_head_df = link_prediction_head_df.dropna().drop_duplicates().reset_index(drop=True)
	return link_prediction_head_df


def corrupt_tails(dataset: pd.DataFrame, nodes: pd.Series):
	final_triples = []
	for idx, (head, relation, tail) in dataset.iterrows():
		final_triples.append([head, relation, tail])  # append correct triple
		for new_tail in nodes:
			final_triples.append([head, relation, new_tail])

	link_prediction_tail_df = pd.DataFrame(final_triples, columns=dataset.columns)
	link_prediction_tail_df = pd.concat([dataset, link_prediction_tail_df], axis=0)
	link_prediction_tail_df = link_prediction_tail_df.dropna().drop_duplicates().reset_index(drop=True)
	return link_prediction_tail_df


def get_scores(model, factory):
	scores = predict_triples(model=model, triples=factory.mapped_triples, mode=None).process(factory)
	scores = scores.df.sort_values(by='score', ascending=False)[['score']].values
	return scores


def get_center(real_train_scores):
	return float(np.mean(real_train_scores))
