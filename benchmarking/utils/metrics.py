import pandas as pd


def hits_at_k(true_entities: pd.DataFrame, predicted_entities: pd.DataFrame, k: int = 1):
	if true_entities.values[0].tolist() in predicted_entities[:k][
		['head_label', 'relation_label', 'tail_label']].values:
		return 1
	return 0


def mean_rank(predicted: pd.DataFrame, true_entities: pd.DataFrame):
	head, relation, tail = true_entities.values[0]

	# get index
	rank = predicted.index[(predicted['head_label'] == head) & (predicted['relation_label'] == relation) & (
			predicted['tail_label'] == tail)].tolist()

	assert len(rank) == 1
	return rank[0] + 1


def mean_reciprocal_rank(true_entities: pd.DataFrame, predicted_entities: pd.DataFrame):
	# real query
	head, relation, tail = true_entities.values[0]

	# Find the rank of the first correctly ranked item (reciprocal rank)
	rank = predicted_entities.index[(
			(predicted_entities['head_label'] == head) & (predicted_entities['relation_label'] == relation) & (
			predicted_entities['tail_label'] == tail))].tolist()

	assert len(rank) == 1
	return 1 / (rank[0] + 1)
