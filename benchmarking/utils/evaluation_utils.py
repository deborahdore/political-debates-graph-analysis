import numpy as np
import pandas as pd
import torch
from pykeen.predict import predict_triples
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer


def get_scores(model, factory, mapped_triples):
	scores = predict_triples(model=model, triples=factory, triples_factory=mapped_triples, mode=None).process(factory)
	scores = scores.df[['score']].values
	return np.sort(scores.ravel())


def get_center(real_train_scores):
	return float(np.mean(real_train_scores))


def get_probabilities_bert(model: BertForSequenceClassification, dataloader: DataLoader, device: torch.device):
	score_test = []
	with torch.no_grad():
		for batch in dataloader:
			batch = tuple(b.to(device) for b in batch)
			inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
			outputs = model(**inputs)
			probs = F.softmax(outputs.logits, dim=-1).detach().cpu()
			max_values, _ = torch.max(probs, dim=-1)
			score_test.append(max_values.numpy())
	return sorted(np.ravel(score_test))


def tokenize_and_generate_dataset(dataset: pd.DataFrame):
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


def adjust_dataset_for_bert(dataset: pd.DataFrame, label:float):
	# create dataset
	new_dataset = pd.DataFrame()
	new_dataset['merged_sent'] = dataset['subject'] + "<sep>" + dataset['predicate'] + "<sep>" + dataset['object']
	new_dataset['label'] = label

	return new_dataset.reset_index(drop=True)
