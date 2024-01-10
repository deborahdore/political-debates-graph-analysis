from typing import Any

import numpy as np
import pandas as pd
import torch
from pykeen.models.predict import predict_triples_df
from pykeen.triples import TriplesFactory
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer

import config


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

	scores = \
	predict_triples_df(model=model, triples=mapped_triples_tensor, triples_factory=None, batch_size=None, mode=None, )[
		"score"].values

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


def get_probabilities_bert(model: BertForSequenceClassification, dataloader: DataLoader, sort: bool = False):
	"""
	Wrapper function for __get_probabilities_bert that returns the scores

	:param model: BertForSequenceClassification: model that will be used
	:param dataloader: DataLoader: dataset to evaluate
	:param sort: bool: if to sort the results or not
	:return: the scores of the evaluated dataset
	"""
	prob, index = __get_probabilities_bert(model, dataloader, sort)
	return prob


def get_probabilities_bert_index(model: BertForSequenceClassification, dataloader: DataLoader, sort: bool = False):
	"""
	Wrapper function for __get_probabilities_bert that returns the corresponding index of the best scores

	:param model: BertForSequenceClassification: model that will be used
	:param dataloader: DataLoader: dataset to evaluate
	:param sort: bool: if to sort the results or not
	:return: the index of the scores of the evaluated dataset
	"""
	prob, index = __get_probabilities_bert(model, dataloader, sort)
	return index


def __get_probabilities_bert(model: BertForSequenceClassification, dataloader: DataLoader, sort: bool = False):
	"""
	Calculates the probability of a triple using bert

	:param model: BertForSequenceClassification: model that will be used
	:param dataloader: DataLoader: dataset to evaluate
	:param sort: bool: if to sort the results or not
	:return: tuple: best score and corresponding index
	"""
	score_test = []
	score_test_index = []
	with torch.no_grad():
		for batch in dataloader:
			batch = tuple(b.to(config.DEVICE) for b in batch)
			inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
			outputs = model(**inputs)
			probs = F.softmax(outputs.logits, dim=-1).detach().cpu()
			max_values, index = torch.max(probs, dim=-1)
			score_test.append(max_values.item())
			score_test_index.append(index.item())

	if sort:
		return sorted(score_test), sorted(score_test_index)
	return score_test, score_test_index


def tokenize_and_generate_dataset(dataset: pd.DataFrame):
	"""
	Prepare the dataset (tokenize and encode) for bert

	:param dataset: DataFrame: dataset to prepare
	:return: the dataset ready to be given to bert
	"""
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	# create bert embeddings
	encoded_data = tokenizer.batch_encode_plus(dataset['merged_sent'],
											   add_special_tokens=True,
											   return_attention_mask=True,
											   padding='max_length',
											   max_length=256,
											   return_tensors='pt')

	dataset = TensorDataset(encoded_data['input_ids'],
							encoded_data['attention_mask'],
							torch.tensor(dataset.label.values))

	return dataset


def adjust_dataset_for_bert(dataset: pd.DataFrame, label: float):
	"""
	Modify the dataset so it fits bert's requirements

	:param dataset: DataFrame: dataset to prepare
	:param label: float: label to assign to each entry
	:return: the dataset ready to be given to bert
	"""
	# create dataset
	new_dataset = pd.DataFrame()
	new_dataset['merged_sent'] = dataset['head'] + "<sep>" + dataset['relation'] + "<sep>" + dataset['tail']
	new_dataset['label'] = label

	return new_dataset.reset_index(drop=True)
