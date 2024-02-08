from typing import Any

import config
import numpy as np
import torch
from pykeen.models.predict import predict_triples_df
from pykeen.triples import TriplesFactory


def get_scores(model: Any, factory: TriplesFactory):
	"""
	The get_scores function takes a model and a factory as input, and returns the scores of all triples in the factory.

	:param model: the model to be used for prediction
	:param factory: the triples for which we want to compute scores
	:return: A numpy array of scores
	"""
	factory.mapped_triples = factory.mapped_triples.to(config.DEVICE)
	scores = predict_triples_df(model=model, triples=factory.mapped_triples, batch_size=None, mode=None)
	return np.sort(scores["score"].values)


def get_scores_tensor(model: Any,
					  triples: [str],
					  entities_label_id_map: {},
					  relation_label_id_map: {},
					  sort: bool = False):
	"""
	The get_scores_tensor function takes a model, triples, and maps of entities and relations to IDs.
	It then returns the scores for each triple in the triples list.


	:param model: Any: the model to be used
	:param triples: [str]: the triples that will be used to calculate the score
	:param entities_label_id_map: {}: Map the entities to their ids
	:param relation_label_id_map: {}: Map the relation label to an id
	:param sort: bool: Sort the scores in ascending order
	:return: The scores of the triples
	"""
	mapped_triples = []
	for h, r, t in triples:
		h_id = entities_label_id_map[h]
		r_id = relation_label_id_map[r]
		t_id = entities_label_id_map[t]
		mapped_triples.append([h_id, r_id, t_id])
	mapped_triples_tensor = torch.tensor(mapped_triples, dtype=torch.long, device=config.DEVICE, requires_grad=False)

	scores = predict_triples_df(model=model,
								triples=mapped_triples_tensor,
								triples_factory=None,
								batch_size=None,
								mode=None, )["score"].values

	assert len(scores) == len(triples)

	if sort:
		return np.sort(scores.ravel())

	return scores.ravel()


def get_center(scores: np.array):
	"""
	The get_center function takes a list of scores and returns the mean of those scores.

	:param scores: []: the scores to be sorted
	:return: The mean of the scores array
	"""
	assert scores.ndim == 1
	return float(np.mean(a=sorted(scores)))  # float(np.median(a=sorted(scores)))
