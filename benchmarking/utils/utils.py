import csv
import json

import pandas as pd
from loguru import logger


def load(file: str) -> []:
	"""
	The load function reads the dataset from a csv file and returns it as a list of lists.
	:return: A list of lists
	"""
	logger.info(f"loading {file}")

	dataset = []
	with open(file, 'r') as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			dataset.append(row)

	return dataset


def save(dataset: [], csv_file: str) -> None:
	"""
	The save function takes a dataset and saves it to a csv file.

	:param dataset: []: Pass in the dataset to be saved
	:param csv_file: str: Specify the file name of the csv file to be saved
	"""
	logger.info(f"saving dataset to {csv_file}")
	with open(csv_file, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(dataset)


def save_tsv(df: pd.DataFrame, tsv_file_path: str) -> None:
	"""
	The save_tsv function takes a pandas dataframe and saves it as a tsv file.
	:param df: pd.DataFrame: Specify the dataframe that is being saved
	:param tsv_file_path: str: Specify the path to the file that will be saved
	:return: None
	"""
	logger.info(f"saving dataframe to tsv {tsv_file_path}")
	df.to_csv(tsv_file_path, sep='\t', index=False, header=True)


def read_json(json_file_name: str) -> json:
	"""
	The read_json function reads a json file and returns the data as a json object.
	:param json_file_name: str: Specify the name of the json file to be read
	:return: A json object
	"""
	logger.info(f"reading json {json_file_name}")
	with open(json_file_name, 'r') as file:
		data = json.load(file)
	file.close()
	return data


def write_json(json_obj: json, json_file_name: str) -> None:
	"""
	The write_json function takes a json object and writes it to a file.

	:param json_obj: json: Specify the type of object that is being passed into the function
	:param json_file_name: str: Specify the name of the file to write
	:return: None
	"""
	logger.info("[write_json] writing json file")
	with open(json_file_name, 'w') as file:
		file.write(json.dumps(json_obj, indent=4))
	file.close()
