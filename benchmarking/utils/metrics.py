import pandas as pd


def hits_at_k_head(true_entities: pd.DataFrame, predicted_entities: pd.DataFrame, k: int = 1):
	hits_count = 0
	for idx, (head, rel, tail) in true_entities.iterrows():
		if [head, rel, tail] in predicted_entities[:k][['head_label', 'relation_label', 'tail_label']].values:
			hits_count += 1

	return hits_count


def mean_rank_head(true_entities: pd.DataFrame, predicted_entities: pd.DataFrame):
	total_rank = 0

	for idx, (head, rel, tail) in true_entities.iterrows():
		# get index
		rank = predicted_entities.tolist().index([head, rel, tail]) + 1
		total_rank += rank

	return total_rank


def mean_reciprocal_rank(true_entities, predicted_entities):
	total_rr = 0
	num_queries = len(true_entities)

	for i in range(num_queries):
		query_true = true_entities[i]
		query_predicted = predicted_entities[i]

		# Find the rank of the first correctly ranked item (reciprocal rank)
		rr = 0
		for j, entity in enumerate(query_predicted):
			if entity in query_true:
				rr = 1 / (j + 1)  # Reciprocal rank formula
				break

		total_rr += rr

	return total_rr
