import numpy as np
from pykeen.predict import predict_triples


def get_scores(model, factory):
	scores = predict_triples(model=model, triples=factory.mapped_triples, mode=None).process(factory)
	scores = scores.df.sort_values(by='score', ascending=False)[['score']].values
	return scores


def get_center(real_train_scores):
	return float(np.mean(real_train_scores))
