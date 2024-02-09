from typing import Any

import numpy as np
import pykeen.models
import torch
from pykeen.models.predict import predict_triples_df
from pykeen.triples import TriplesFactory

from benchmarking import config


def get_scores(model: pykeen.models.Model, factory: TriplesFactory):
	""" Computes the scores of all triples in the factory """
	factory.mapped_triples = factory.mapped_triples.to(config.DEVICE)
	scores = predict_triples_df(model=model, triples=factory.mapped_triples, batch_size=None, mode=None)
	return np.sort(scores["score"].values)


def get_scores_tensor(model: Any,
					  triples: [str],
					  entities_label_id_map: {},
					  relation_label_id_map: {},
					  sort: bool = False):
	""" Computes the scores for each triple in the triples list"""

	mapped_triples = []
	for h, r, t in triples:
		h_id = entities_label_id_map.get(h, len(entities_label_id_map.keys()))
		r_id = relation_label_id_map.get(r, len(entities_label_id_map.keys()))
		t_id = entities_label_id_map.get(t, len(entities_label_id_map.keys()))
		mapped_triples.append([h_id, r_id, t_id])
	mapped_triples_tensor = torch.tensor(mapped_triples, dtype=torch.long, device=config.DEVICE, requires_grad=False)

	try:
		scores = predict_triples_df(model=model,
									triples=mapped_triples_tensor,
									triples_factory=None,
									batch_size=None,
									mode=None, )["score"].values
	except Exception as err:
		print(err)
		return

	assert len(scores) == len(triples)

	if sort:
		return np.sort(scores.ravel())

	return scores.ravel()


def get_center(scores: np.array):
	"""  Computes the mean of scores"""
	assert scores.ndim == 1
	return float(np.mean(a=sorted(scores)))  # float(np.median(a=sorted(scores)))
