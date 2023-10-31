import math
import random

import pandas as pd
import torch
from loguru import logger
from pykeen.triples import TriplesFactory
from sklearn.model_selection import train_test_split
from utils.utils import load, read_json, read_tsv, save, save_json

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_nodes(dataset: pd.DataFrame):
	"""
	The get_nodes function takes a dataset as input and returns the unique nodes in that dataset.

	:param dataset: pd.DataFrame: Extract the nodes from the dataset
	:return: A list of all unique nodes in the dataset
	"""
	logger.info("extracting nodes from dataset")

	nodes = pd.concat([dataset['head'], dataset['tail']], axis=0)
	nodes = nodes.sample(frac=1)  # shuffle
	nodes = nodes.dropna().drop_duplicates().reset_index(drop=True)
	return nodes


def generate_triplets(original_dataset_file: str, triples_file: str):
	"""
	The generate_triplets function takes in a file path to an original dataset and a file path to the destination
	file where the generated triples will be saved. The function then loads the original dataset, creates nodes for
	the subject and object of each triplet, drops unnecessary columns from the dataframe, renames columns so that they
	match what is expected by our model (subject, predicate, object), preprocesses created dataset by converting all
	strings to lowercase and removing duplicates/null values. Finally it saves this new dataframe as a list of lists in
	the destination file.

	:param original_dataset_file: str: Specify the file path of the original dataset
	:param triples_file: str: Specify the file that contains the triples
	:return: A list of triples
	"""
	logger.info("generating triplets")
	logger.info(f"original dataset: {original_dataset_file}")
	logger.info(f"destination file: {triples_file}")

	# load original dataset
	original_dataset = load(original_dataset_file)

	# keep only useful columns
	df = pd.DataFrame(original_dataset[1:], columns=original_dataset[0])[
		['Dependent', 'D_type', 'Governor', 'G_type', 'RelationType']]

	# create nodes -> Governor = head, Dependent = tail, RelationType = relation
	df['head'] = df.apply(lambda row: str(row['Governor']) + "-" + str(row['G_type']), axis=1)
	df['tail'] = df.apply(lambda row: str(row['Dependent']) + "-" + str(row['D_type']), axis=1)

	df.drop(columns=['Dependent', 'D_type', 'Governor', 'G_type'], inplace=True)
	df = df[['head', 'RelationType', 'tail']]
	df.columns = ['head', 'relation', 'tail']

	logger.info("preprocessing created dataset")
	df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
	df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
	df = df.dropna().drop_duplicates().reset_index(drop=True)

	columns = df.columns
	save([columns] + df.values.tolist(), triples_file)


def generate_noise(triplets_file: str, noisy_triples_file: str, valid_noise: [int]):
	"""
	The generate_noise function takes in a triplets file and generates noisy triples.
	The function will generate the following files:
		- train_noisy_{noise}.csv, test_noisy_{noise}.csv, val_noisy_{noise}.csv for each noise ratio in valid_ratio.

	:param triplets_file: str: Specify the path to the file containing all triplets
	:param noisy_triples_file: str: Specify the file path of the noisy triples
	:param valid_noise: [float]: list of the noise ratio
	:return: A dataframe with the following columns: head, relation, tail, noise (1 if it's a synthetic noisy triple,
	0 if not)
	"""
	for noise_ratio in valid_noise:
		df = __generate_noise(triplets_file, noise_ratio=noise_ratio)

		# split dataset in partitions [0.8, 0.1, 0.1]
		train, test = train_test_split(df, test_size=0.1, random_state=42, stratify=df['noise'])
		train, val = train_test_split(train, test_size=0.1, random_state=42, stratify=train['noise'])

		columns = df.columns
		save([columns] + train.values.tolist(), noisy_triples_file.format(noise=noise_ratio, use="train"))
		save([columns] + test.values.tolist(), noisy_triples_file.format(noise=noise_ratio, use="test"))
		save([columns] + val.values.tolist(), noisy_triples_file.format(noise=noise_ratio, use="val"))


def __generate_noise(triplets_file: str, noise_ratio: float):
	"""
	Helper function that generates noise

	:param triplets_file: str: Specify the path to the file that contains all of the triplets
	:param noise_ratio: float: Determine the amount of noise to be added to the dataset
	:return: A dataframe with the original triplets and noise triplets
	"""
	logger.info(f"generating dataset with {noise_ratio}% noise")
	logger.info(f"source file: {triplets_file}")

	# loading real triples
	edges_correct = load(triplets_file)
	edges_correct = pd.DataFrame(edges_correct[1:], columns=edges_correct[0])

	# ------ special case : leave original dataset as it is ------ #
	if noise_ratio == 0.0:
		edges_correct['noise'] = 0
		return edges_correct

	# ------ special case : half good half fake ------ #
	if noise_ratio == 100:
		noise_to_add = len(edges_correct)
	else:
		noise_to_add = int(math.ceil((noise_ratio / 100) * len(edges_correct)))

	edges_noisy = []
	nodes = get_nodes(edges_correct)
	relations = ["attack", "support", "equivalent"]

	for _ in range(noise_to_add):
		head = random.choice(nodes)
		rel = random.choice(relations)
		tail = random.choice(nodes)

		while [head, rel, tail] in edges_correct.values.tolist():
			# avoid creating fake triple that already exists in list of correct ones
			head = random.choice(nodes)
			rel = random.choice(relations)
			tail = random.choice(nodes)

		edges_noisy.append([head, rel, tail])

	edges_noisy = pd.DataFrame(edges_noisy, columns=['head', 'relation', 'tail'])
	edges_noisy = edges_noisy.dropna()

	edges_correct['noise'] = 0
	edges_noisy['noise'] = 1  # identify noisy triples

	final_df = pd.concat([edges_correct, edges_noisy], axis=0).sample(frac=1)
	final_df = final_df.dropna().drop_duplicates().reset_index(drop=True)

	return final_df


def get_mapping(dataset: pd.DataFrame):
	"""
	The get_mapping function takes a dataset as input and returns two dictionaries:
		- entity_to_id: maps each entity to an integer id.
		- relation_to_id: maps each relation to an integer id.

	:param dataset: pd.DataFrame: Get the mapping from entities to ids and relations to ids
	:return: A mapping from entities to ids and a mapping from relations to ids
	"""
	dataset = TriplesFactory.from_labeled_triples(triples=dataset[['head', 'relation', 'tail']].values,
												  create_inverse_triples=False)
	return dataset.entity_to_id, dataset.relation_to_id


def get_train_val_test_factory(train: pd.DataFrame,
							   val: pd.DataFrame,
							   test: pd.DataFrame,
							   triplets_file_utils: str,
							   create_inverse_triples: bool):
	"""
	The get_train_val_test_factory function creates a TriplesFactory object for each of the train, val and test sets.

	:param train: pd.DataFrame: A pandas DataFrame containing the training triples.
	:param val: pd.DataFrame: A pandas DataFrame containing the validation triples.
	:param test: pd.DataFrame: A pandas DataFrame containing the testing triples.
	:param triplets_file_utils: str: Specify the location of the json files containing entity_to_id and relation_to_id
	:param create_inverse_triples: bool: Create inverse triples
	:return: The train, validation and test data as triplesfactory objects
	"""
	logger.info("creating train, test and val TriplesFactory")

	entity_to_id = read_json(triplets_file_utils.format(file_name="entity_to_id"))
	relation_to_id = read_json(triplets_file_utils.format(file_name="relation_to_id"))

	train_factory = TriplesFactory.from_labeled_triples(triples=train[['head', 'relation', 'tail']].values,
														entity_to_id=entity_to_id,
														relation_to_id=relation_to_id,
														create_inverse_triples=create_inverse_triples)

	val_factory = TriplesFactory.from_labeled_triples(triples=val[['head', 'relation', 'tail']].values,
													  entity_to_id=entity_to_id,
													  relation_to_id=relation_to_id,
													  create_inverse_triples=create_inverse_triples)

	test_factory = TriplesFactory.from_labeled_triples(triples=test[['head', 'relation', 'tail']].values,
													   entity_to_id=entity_to_id,
													   relation_to_id=relation_to_id,
													   create_inverse_triples=create_inverse_triples)

	return train_factory, val_factory, test_factory


def get_factory(dataset: pd.DataFrame, entity_to_id: dict, relation_to_id: dict, create_inverse_triples: bool = False):
	"""
	The get_factory function takes in a dataset, an entity_to_id dictionary, and a relation_to_id dictionary.
	It then creates the TriplesFactory object from the labeled triples of the dataset.
	The function also has an optional parameter create_inverse_triples which is set to False by default.

	:param dataset: pd.DataFrame: Create the triplesfactory object
	:param entity_to_id: dict: Map the entity to a unique id
	:param relation_to_id: dict: Map the relation to an integer
	:param create_inverse_triples: bool: Create inverse triples
	:return: A triplesfactory object
	"""
	factory = TriplesFactory.from_labeled_triples(triples=dataset[['head', 'relation', 'tail']].values,
												  entity_to_id=entity_to_id,
												  relation_to_id=relation_to_id,
												  create_inverse_triples=create_inverse_triples)
	return factory


def get_train_val_test_from_dir(noisy_triples_file: str,
								noise: float,
								drop_col_noise: bool = True,
								get_noisy_test: bool = False):
	"""
	The get_train_val_test_from_dir function takes in a noisy_triples_file, which is the path to a file containing
	noisy triples. The function then loads the train, val and test sets from this file. It returns these three sets as
	pandas dataframes.

	:param noisy_triples_file: str: Specify the file that contains the noisy triples
	:param noise: float: Specify the amount of noise in the data
	:param drop_col_noise: bool: Drop the noisy column from the dataframe
	:param get_noisy_test: bool: specify if to return test with noise or not
	:return: A tuple of dataframes, one for each dataset
	"""
	train = load(noisy_triples_file.format(use="train", noise=noise))
	train = pd.DataFrame(data=train[1:], columns=train[0])

	val = load(noisy_triples_file.format(use="val", noise=noise))
	val = pd.DataFrame(data=val[1:], columns=val[0])

	if get_noisy_test:
		test = load(noisy_triples_file.format(use="test", noise=noise))
	else:
		test = load(noisy_triples_file.format(use="test", noise=0))

	test = pd.DataFrame(data=test[1:], columns=test[0])

	if drop_col_noise:
		train = train.drop("noise", axis=1)
		val = val.drop("noise", axis=1)
		test = test.drop("noise", axis=1)

	return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def generate_mappings(triplets_file: str, triplets_file_utils: str):
	"""
	The generate_mappings function takes in a triplets file and a triplets_file_utils string.
	The function reads the triplet file, creates an entity to id mapping and relation to id mapping,
	and saves them as json files into the triplets_file_utils file.

	:param triplets_file: str: Specify the file path of the triplets
	:param triplets_file_utils: str: Specify the path to the directory where you want to save your entity_to_id and
	relation_to_id mapping
	"""
	logger.info("creating entity-to-id and relation-to-id mappings")

	# read original tsv file
	triplets = read_tsv(triplets_file)
	# generate mappings using pykeen function
	factory = TriplesFactory.from_labeled_triples(triples=triplets[['head', 'relation', 'tail']].values,
												  create_inverse_triples=False)

	# save mappings
	entity_to_id = factory.entity_to_id
	relation_to_id = factory.relation_to_id

	entity_to_id_file = triplets_file_utils.format(file_name="entity_to_id")
	save_json(entity_to_id, entity_to_id_file)

	relation_to_id_file = triplets_file_utils.format(file_name="relation_to_id")
	save_json(relation_to_id, relation_to_id_file)
