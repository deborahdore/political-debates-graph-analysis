import random

import pandas as pd


def corrupt_head(dataset: pd.DataFrame, nodes: pd.Series):
	final_triples = []
	for idx, (head, relation, tail) in dataset.iterrows():
		new_head = random.choice(nodes)
		while new_head == head:
			new_head = random.choice(nodes)
		final_triples.append([new_head, relation, tail])

	corrupted_dataset_head = pd.DataFrame(final_triples, columns=dataset.columns)
	corrupted_dataset_head = corrupted_dataset_head.dropna().drop_duplicates().reset_index(drop=True)
	return corrupted_dataset_head


def corrupt_tail(dataset: pd.DataFrame, nodes: pd.Series):
	final_triples = []
	for idx, (head, relation, tail) in dataset.iterrows():
		new_tail = random.choice(nodes)
		while new_tail == tail:
			new_tail = random.choice(nodes)
		final_triples.append([head, relation, new_tail])

	corrupted_dataset_head = pd.DataFrame(final_triples, columns=dataset.columns)
	corrupted_dataset_head = corrupted_dataset_head.dropna().drop_duplicates().reset_index(drop=True)
	return corrupted_dataset_head


def corrupt_heads(dataset: pd.DataFrame, nodes: pd.Series):
	final_triples = []
	for idx, (head, relation, tail) in dataset.iterrows():
		for new_head in nodes:
			final_triples.append([new_head, relation, tail])

	link_prediction_head_df = pd.DataFrame(final_triples, columns=dataset.columns)
	link_prediction_head_df = pd.concat([dataset, link_prediction_head_df], axis=0)
	link_prediction_head_df = link_prediction_head_df.dropna().drop_duplicates().reset_index(drop=True)
	return link_prediction_head_df


def corrupt_tails(dataset: pd.DataFrame, nodes: pd.Series):
	final_triples = []
	for idx, (head, relation, tail) in dataset.iterrows():
		for new_tail in nodes:
			final_triples.append([head, relation, new_tail])

	link_prediction_tail_df = pd.DataFrame(final_triples, columns=dataset.columns)
	link_prediction_tail_df = pd.concat([dataset, link_prediction_tail_df], axis=0)
	link_prediction_tail_df = link_prediction_tail_df.dropna().drop_duplicates().reset_index(drop=True)
	return link_prediction_tail_df
