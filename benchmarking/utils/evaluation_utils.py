import random

import pandas as pd

from benchmarking.utils.dataset_utils import generate_noise, get_factory
from benchmarking.utils.utils import load


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


def load_and_prepare_for_triple_classification(file_name: str, use: str, noisy: bool = False):
	if not noisy:
		data = load(file_name.format(use=use))
		data_df = pd.DataFrame(data[1:], columns=data[0])
	else:
		data_df = generate_noise(file_name, noise_ratio=1, use=use)

	factory = get_factory(data_df)
	return data_df, factory
