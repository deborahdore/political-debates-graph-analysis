import csv
import json

import pandas as pd
import torch
from loguru import logger


def load(file: str):
	"""
	The load function takes a file name as an argument and returns the contents of that file in a list.
	The function opens the specified file, reads it line by line, and appends each row to a list.


	:param file: str: Specify the file name to be loaded
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
	The save function takes a dataset and saves it to a csv file.

	:param dataset: []: Specify the dataset to be saved
	:param csv_file: str: Specify the file name of the csv to be saved
	:return: The number of rows written to the csv file
	"""
	logger.info(f"saving dataset to {csv_file}")
	with open(csv_file, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(dataset)


def save_tsv(df: pd.DataFrame, tsv_file_path: str):
	"""
	The save_tsv function saves a dataframe to a tsv file.

	:param df: pd.DataFrame: Specify the dataframe that is being passed into the function
	:param tsv_file_path: str: Specify the path to the tsv file
	:return: A dataframe that is saved as a tsv file
	"""
	logger.info(f"saving dataframe to tsv {tsv_file_path}")
	df.to_csv(tsv_file_path, sep='\t', index=False, header=True)


def read_tsv(tsv_file_path: str):
	"""
	The read_tsv function reads in a tab-separated file and returns a pandas dataframe.

	:param tsv_file_path: str: Specify the path of the file to be loaded
	:return: A pandas dataframe
	"""
	logger.info(f"Loading {tsv_file_path}")
	return pd.read_csv(tsv_file_path, delimiter="\t")


def read_json(json_file_name: str):
	"""
	The read_json function reads a json file and returns the data as a dictionary.

	:param json_file_name: str: Specify the file name of the json file that will be read
	:return: A dictionary
	"""
	logger.info(f"reading json {json_file_name}")
	with open(json_file_name, 'r') as file:
		data = json.load(file)
	file.close()
	return data


def save_json(json_obj: json, json_file_name: str):
	"""
	The save_json function takes a json object and saves it to the specified file name.


	:param json_obj: json: Specify the type of data that is being passed into the function
	:param json_file_name: str: Specify the name of the json file to be written
	:return: Nothing
	"""
	logger.info("[write_json] writing json file")
	with open(json_file_name, 'w') as file:
		file.write(json.dumps(json_obj, indent=4))
	file.close()


def load_model(model_file: str, device: torch.device):
	"""
	The load_model function loads a model from the specified file and returns it.

	:param model_file: str: Specify the path to the model file
	:param device:torch.device: Specify the device on which to load the model
	:return: A dictionary of the model's state_dict
	"""
	return torch.load(model_file, map_location=device)
