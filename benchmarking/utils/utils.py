import csv
import json

import pandas as pd
import torch
from loguru import logger


def load(file: str):
	logger.info(f"loading {file}")
	dataset = []
	with open(file, 'r') as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			dataset.append(row)

	return dataset


def save(dataset: [], csv_file: str):
	logger.info(f"saving dataset to {csv_file}")
	with open(csv_file, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(dataset)


def save_tsv(df: pd.DataFrame, tsv_file_path: str):
	logger.info(f"saving dataframe to tsv {tsv_file_path}")
	df.to_csv(tsv_file_path, sep='\t', index=False, header=True)


def read_tsv(tsv_file_path: str):
	logger.info(f"Loading {tsv_file_path}")
	return pd.read_csv(tsv_file_path, index_col=False, header=0)


def read_json(json_file_name: str):
	logger.info(f"reading json {json_file_name}")
	with open(json_file_name, 'r') as file:
		data = json.load(file)
	file.close()
	return data


def save_json(json_obj: json, json_file_name: str):
	logger.info("[write_json] writing json file")
	with open(json_file_name, 'w') as file:
		file.write(json.dumps(json_obj, indent=4))
	file.close()


def load_model(model_file: str, device:torch.device):
	return torch.load(model_file, map_location=device)
