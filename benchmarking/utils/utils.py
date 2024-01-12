import csv
import json

import config
import pandas as pd
import torch
from loguru import logger


def load(file: str):
	"""
	Load a file

	:param file: str: file path
	:return: A list of lists
	"""
	logger.info(f"loading {file}")
	dataset = []
	with open(file, 'r') as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			dataset.append(row)

	return dataset


def save(dataset: [], csv_file: str):
	"""
	Saves into a csv

	:param dataset: []: dataset to be saved
	:param csv_file: str: file where dataset will be saves
	"""
	logger.info(f"saving dataset to {csv_file}")
	with open(csv_file, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(dataset)


def save_tsv(df: pd.DataFrame, tsv_file_path: str):
	"""
	Saves dataframe into a tsv

	:param df: pd.Dataframe: dataframe to be saved
	:param tsv_file_path: str: file where the dataframe will be saved
	"""
	logger.info(f"saving dataframe to tsv {tsv_file_path}")
	df.to_csv(tsv_file_path, index=False, header=True)


def read_tsv(tsv_file_path: str):
	"""
	Reads a tsv into a dataframe

	:param tsv_file_path: str: file to be read
	:return: A pandas dataframe
	"""
	logger.info(f"Loading {tsv_file_path}")
	return pd.read_csv(tsv_file_path, index_col=False, header=0)


def read_json(json_file_name: str):
	"""
	Read a json file

	:param json_file_name: str: json file that will be read
	:return: A dictionary
	"""
	logger.info(f"reading json {json_file_name}")
	with open(json_file_name, 'r') as file:
		data = json.load(file)
	file.close()
	return data


def save_json(json_obj: json, json_file_name: str):
	"""
	Saves json

	:param json_obj: json: json object to save
	:param json_file_name: str: file where the json object will be saved
	"""
	logger.info("[write_json] writing json file")
	with open(json_file_name, 'w') as file:
		file.write(json.dumps(json_obj, indent=4))
	file.close()


def load_model(model_file: str):
	"""
	Load a torch model

	:param model_file: str: file that contains the model parameters
	:return: the model
	"""
	return torch.load(model_file, map_location=config.DEVICE)
