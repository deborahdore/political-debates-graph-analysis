import csv
import json

import pandas as pd
import torch

from benchmarking import config


def read_csv(tsv_file_path: str):
	"""
	Reads a tsv into a dataframe

	:param tsv_file_path: str: file to be read
	:return: A pandas dataframe
	"""
	return preprocess_dataset(pd.read_csv(tsv_file_path, index_col=False, header=0))


def save_csv(dataframe: pd.DataFrame, tsv_file_path: str):
	"""
	Reads a tsv into a dataframe

	:param tsv_file_path: str: file where to save the df
	:param dataframe: pd.DataFrame: obj to save
	"""
	return preprocess_dataset(dataframe).to_csv(tsv_file_path, index=False)


def save(dataset: [], csv_file: str):
	"""
	Saves into a csv

	:param dataset: []: dataset to be saved
	:param csv_file: str: file where dataset will be saves
	"""
	with open(csv_file, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(dataset)


def save_tsv(df: pd.DataFrame, tsv_file_path: str):
	"""
	Saves dataframe into a tsv

	:param df: pd.Dataframe: dataframe to be saved
	:param tsv_file_path: str: file where the dataframe will be saved
	"""
	preprocess_dataset(df).to_csv(tsv_file_path, index=False, sep="\t")


def read_tsv(tsv_file_path: str):
	"""
	Reads a tsv into a dataframe

	:param tsv_file_path: str: file to be read
	:return: A pandas dataframe
	"""
	return preprocess_dataset(pd.read_csv(tsv_file_path, index_col=False, header=0, sep="\t"))


def read_json(json_file_name: str):
	"""
	Read a json file

	:param json_file_name: str: json file that will be read
	:return: A dictionary
	"""
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


def preprocess_dataset(df: pd.DataFrame):
	"""
	Processes a dataframe

	:param df: DataFrame: the dataframe to process
	:return: the processed dataframe
	"""
	df = df.applymap(lambda x: str(x))
	df = df.applymap(lambda x: x.lower())
	df = df.applymap(lambda x: x.strip("."))
	df = df.applymap(lambda x: x.strip())
	df = df.dropna().drop_duplicates().reset_index(drop=True)
	return df
