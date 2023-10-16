import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer


def adjust_dataset_for_bert(dataset: pd.DataFrame):
	# create dataset
	new_dataset = pd.DataFrame()
	new_dataset['merged_sent'] = dataset['subject'] + "<sep>" + dataset['object']
	new_dataset['label'] = dataset['predicate'].replace({'support': 0, 'attack': 1, 'equivalent': 2})
	return new_dataset.reset_index(drop=True)


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


def get_probabilities_bert(model: BertForSequenceClassification, dataloader: DataLoader, device: str):
	score_test = []
	with torch.no_grad():
		for batch in dataloader:
			batch = tuple(b.to(device) for b in batch)
			inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
			outputs = model(**inputs)
			score_test.append(F.softmax(outputs.logits.detach().cpu().numpy(), axis=-1))

	return sorted(score_test)
