import os

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

from benchmarking.utils.utils import save

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

bert_without_components = "data/bert_without_argument_component.csv"
bert_with_components = "data/bert_with_argument_component.csv"
bert_dir = "../dataset/bert"


class MyDataset(torch.utils.data.Dataset):
	def __init__(self, encodings, labels, subject, object1):
		self.subjects = subject
		self.objects = object1
		self.encodings = encodings
		self.labels = labels

	def __getitem__(self, idx):
		item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		item['labels'] = torch.tensor(self.labels[idx])
		item['subject'] = self.subjects[idx]
		item['object'] = self.objects[idx]
		return item

	def __len__(self):
		return len(self.labels)


def bert_processing():
	df = pd.read_csv(bert_with_components)

	# create triples file
	triples = pd.DataFrame()
	triples['subject'] = df['subject']
	triples['object'] = df['object']
	triples['predicate'] = df['predicted']
	triples['noisy'] = (df['relation'] == df['predicted']).astype(int)

	triples['predicate'] = triples['predicate'].replace({1: 'Support', 0: 'Attack'})

	triples = triples.map(lambda x: x.lower() if isinstance(x, str) else x)

	# split
	train, test = train_test_split(triples, test_size=0.2, random_state=42, stratify=triples['noisy'])
	train, val = train_test_split(train, test_size=0.15, random_state=42, stratify=train['noisy'])

	columns = triples.columns
	save([columns] + train.values.tolist(), os.path.join(bert_dir, "triplets_file_train.tsv"))
	save([columns] + test.values.tolist(), os.path.join(bert_dir, "triplets_file_test.tsv"))
	save([columns] + val.values.tolist(), os.path.join(bert_dir, "triplets_file_val.tsv"))


def bert_training():
	# process original dataset to adapt it to bert
	original_dataset = pd.read_csv('/content/original_dataset.csv')
	original_dataset['label'] = original_dataset['RelationType']
	original_dataset['label'] = original_dataset['label'].replace({'Support': 1, 'Attack': 0, 'Equivalent': -1})
	original_dataset = original_dataset[original_dataset['label'] != -1]
	original_dataset = original_dataset.drop(columns=['Year', 'date', 'Speaker1', 'Speaker2', 'long_date'])
	original_dataset['subject'] = original_dataset.apply(lambda row: str(row['Governor']) + "-" + str(row['G_type']),
														 axis=1)
	original_dataset['object'] = original_dataset.apply(lambda row: str(row['Dependent']) + "-" + str(row['D_type']),
														axis=1)
	# define merged sentence -> could be also without argument components
	original_dataset['merged_sent'] = original_dataset['object'] + original_dataset['subject']

	# Load the BERT tokenizer
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	# split dataset
	train_set, test_set = train_test_split(original_dataset, test_size=0.3, random_state=42)

	# create bert embeddings
	embeddings_train = tokenizer(train_set['merged_sent'].tolist(), truncation=True, padding=True)
	embeddings_test = tokenizer(test_set['merged_sent'].tolist(), truncation=True, padding=True)

	# create custom datasets
	train_dataset = MyDataset(embeddings_train,
							  train_set['label'].tolist(),
							  train_set['subject'].tolist(),
							  train_set['object'].tolist())
	test_dataset = MyDataset(embeddings_test,
							 test_set['label'].tolist(),
							 test_set['subject'].tolist(),
							 test_set['object'].tolist())

	# create dataloader
	train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

	# create model
	model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
	model.to(device)
	model.train()

	# define optimizer
	optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

	# train loop  -> 4 epochs
	for epoch in tqdm(range(4)):
		for idx, batch in enumerate(train_loader):
			optim.zero_grad()
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
			outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
			loss = outputs[0]
			loss.backward()
			optim.step()

	# Evaluate the model
	model.eval()
	y_true, y_pred = [], []

	data = []

	for batch in test_loader:
		optim.zero_grad()
		input_ids = batch['input_ids'].to(device)
		attention_mask = batch['attention_mask'].to(device)
		outputs = model(input_ids, attention_mask=attention_mask)
		logits = outputs.logits.detach().cpu().numpy()
		predictions = np.argmax(logits, axis=1)
		y_true.extend(batch['labels'].cpu().numpy())
		y_pred.extend(predictions)

		decoded_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

		for i in range(len(decoded_texts)):
			data.append({
				'subject'  : batch['subject'][i],
				'object'   : batch['object'][i],
				'relation' : batch['labels'][i].item(),
				'predicted': predictions[i]})

	df = pd.DataFrame(data)

	df['relation'] = df['relation'].replace({1: 'Support', 0: 'Attack'})
	df['predicted'] = df['predicted'].replace({1: 'Support', 0: 'Attack'})
	df['noisy'] = (df['relation'] == df['predicted']).astype(int)

	df.to_csv("data/bert_with_argument_component.csv", index=False)

	class_report = classification_report(y_true, y_pred)
	logger.info("Report di classificazione per la prima classificazione:")
	logger.info(class_report)


if __name__ == '__main__':
	bert_training()
	bert_processing()
