import math

import config
import numpy
import pandas as pd
from loguru import logger
from pykeen.triples import TriplesFactory
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from utils.utils import load, read_json, read_tsv, save_json, save_tsv

political_positions = {
	'NIXON'    : 'President',
	'KENNEDY'  : 'President',
	'CARTER'   : 'President',
	'FORD'     : 'President',
	'MONDALE'  : 'Vice-President',
	'BUSH'     : 'President',
	'FERRARO'  : 'Vice-Presidential candidate',
	'DUKAKIS'  : 'Presidential candidate',
	'PEROT'    : 'Presidential candidate',
	'CLINTON'  : 'President',
	'GORE'     : 'Vice-President',
	'QUAYLE'   : 'Vice-President',
	'STOCKDALE': 'Vice-Presidential candidate',
	'DOLE'     : 'Vice-Presidential candidate',
	'KEMP'     : 'Vice-Presidential candidate',
	'LIEBERMAN': 'Vice-Presidential candidate',
	'CHENEY'   : 'Vice-President',
	'EDWARDS'  : 'Vice-Presidential candidate',
	'KERRY'    : 'Presidential candidate',
	'BIDEN'    : 'Vice-President',
	'PALIN'    : 'Vice-Presidential candidate',
	'MCCAIN'   : 'Presidential candidate',
	'OBAMA'    : 'President',
	'ROMNEY'   : 'Presidential candidate',
	'RYAN'     : 'Vice-Presidential candidate',
	'TRUMP'    : 'President',
	'PENCE'    : 'Vice-President'}


def get_nodes(dataset: pd.DataFrame):
	"""
	Finds the unique nodes in the dataset

	:param dataset: pd.DataFrame: the dataset where to search the nodes
	:return: A list of all unique nodes in the dataset
	"""
	logger.info("extracting nodes from dataset")

	nodes = pd.concat([dataset['head'], dataset['tail']], axis=0)
	nodes = nodes.sample(frac=1)  # shuffle
	nodes = nodes.dropna().drop_duplicates().reset_index(drop=True)
	return nodes


def generate_triplets(original_dataset_file: str, triples_file: str, original_triplets_file: str):
	"""
	Generate the triples from the dataset

	:param original_dataset_file: str: the original dataset file
	:param triples_file: str: file used to save the new triples created with the corresponding modalities
	:param original_triplets_file: file used to save the pure triples without any addition
	"""
	logger.info("generating triplets")
	logger.info(f"original dataset: {original_dataset_file}")
	logger.info(f"destination file: {triples_file}")

	# load original dataset
	original_dataset = load(original_dataset_file)

	# keep only useful columns
	df_original = pd.DataFrame(original_dataset[1:], columns=original_dataset[0])[
		['Dependent', 'D_type', 'Speaker1', 'Governor', 'G_type', 'Speaker2', 'RelationType', 'long_date']]

	df = configure_nodes(df_original, config.MODE_TEXT)
	save_tsv(preprocess_dataset(df), original_triplets_file)

	for mode_node in config.MODE_NODE:
		df = pd.concat([df, add_nodes(df_original, mode_node, mode_text=config.MODE_TEXT)], axis=0)
	save_tsv(preprocess_dataset(df), triples_file)


def preprocess_dataset(df: pd.DataFrame):
	"""
	Processes a dataframe

	:param df: DataFrame: the dataframe to process
	:return: the processed dataframe
	"""
	logger.info("preprocessing created dataset")
	df = df.applymap(lambda x: str(x))
	df = df.applymap(lambda x: x.lower())
	df = df.applymap(lambda x: x.strip())
	df = df.applymap(lambda x: x.strip("."))
	df = df.dropna().drop_duplicates().sample(frac=1).reset_index(drop=True)
	return df


def configure_nodes(df: pd.DataFrame, mode: str):
	"""
	Create nodes content with the specified mode.

	:param df: pd.DataFrame: starting dataframe
	:param mode: str: Determine which nodes to add to the dataframe
	:return: a dataframe
	"""
	logger.info(f"Creating nodes content with mode: {mode.upper()}")

	if mode == 'text':
		head = df['Governor'].copy()
		relation = df['RelationType'].copy()
		tail = df['Dependent'].copy()
	elif mode == 'text+speaker':
		head = df.apply(lambda row: str(row['Governor']) + " " + str(row['Speaker2']), axis=1)
		relation = df['RelationType'].copy()
		tail = df.apply(lambda row: str(row['Dependent']) + " " + str(row['Speaker1']), axis=1)
	elif mode == 'text+claim':
		head = df.apply(lambda row: str(row['Governor']) + " " + str(row['G_type']), axis=1)
		relation = df['RelationType'].copy()
		tail = df.apply(lambda row: str(row['Dependent']) + " " + str(row['D_type']), axis=1)
	elif mode == 'text+claim+speaker':
		head = df.apply(lambda row: str(row['Governor']) + " " + str(row['G_type']) + " " + str(row['Speaker2']),
						axis=1)
		relation = df['RelationType'].copy()
		tail = df.apply(lambda row: str(row['Dependent']) + " " + str(row['D_type']) + " " + str(row['Speaker1']),
						axis=1)

	assert head.size == relation.size == tail.size
	new_df = pd.DataFrame({'head': head, 'relation': relation, 'tail': tail})
	new_df = new_df.dropna().reset_index(drop=True)
	return new_df


def add_nodes(df_original: pd.DataFrame, mode: str, mode_text: str):
	"""
	Wrapper function that selects the modality for adding nodes based on the modality that was chosen for the features
	of the nodes before

	:param df_original: pd.DataFrame: starting dataframe
	:param mode: str: the modality that specifies which nodes to add
	:param mode_text: str: determine whether the nodes were created from mode text or mode text+claim
	:return: A dataframe
	"""
	if mode_text == "text":
		df = __add_nodes_mode_text(df_original, mode)
	else:  # text+claim
		df = __add_nodes_mode_claim(df_original, mode)

	return df.dropna().reset_index(drop=True)


def __add_nodes_mode_text(df: pd.DataFrame, mode: str):
	"""
	The __add_nodes_mode_text function creates nodes based on the mode argument.
	The modes are: claim+premise, speaker, and year.
		- If the mode is 'claim+premise', then it will create a node "claim" or "premise" for each node  in the
		dataframe with a relation "is a" to their respective type.
		- If the mode is 'speaker', then it will create a node "speaker" for each node in the dataframe with
		relations of "said by" to their respective speakers (Speaker 1 or Speaker 2) as well as add all political
		positions to the speaker's node
		- If the mode is 'year', then it will create a node "year" with the year in which that debate was held and it
		will connect to the respective nodes in the dataset with the relation "said in"

	:param df: pd.DataFrame: the original dataframe
	:param mode: str: Create the nodes based on the mode
	:return: A dataframe
	"""

	logger.info(f"Creating nodes with mode: {mode.upper()}")  # based on mode nodes "text"
	if mode == 'claim+premise':
		head = pd.concat([df['Dependent'].copy(), df['Governor'].copy()], axis=0)
		tail = pd.concat([df['D_type'].copy(), df['G_type'].copy()], axis=0)
		assert head.size == tail.size
		return pd.DataFrame({'head': head, 'relation': "is a", 'tail': tail})

	elif mode == 'speaker':
		speaker_head = pd.concat([df['Dependent'].copy(), df['Governor'].copy()], axis=0)
		speaker_tail = pd.concat([df['Speaker1'].copy(), df['Speaker2'].copy()], axis=0)

		assert speaker_head.size == speaker_tail.size
		speaker_df = pd.DataFrame({'head': speaker_head, 'relation': "said by", 'tail': speaker_tail})

		role_head = pd.Series(list(political_positions.keys()))
		role_tail = pd.Series(list(political_positions.values()))

		assert role_head.size == role_tail.size
		role_df = pd.DataFrame({'head': role_head, 'relation': "role", 'tail': role_tail})

		return pd.concat([speaker_df, role_df], axis=0)

	elif mode == 'year':
		head = pd.concat([df['Dependent'].copy(), df['Governor'].copy()], axis=0)
		tail = pd.concat([df['long_date'].copy(), df['long_date'].copy()], axis=0)

		assert head.size == tail.size
		return pd.DataFrame({'head': head, 'relation': "said in", 'tail': tail})


def __add_nodes_mode_claim(df: pd.DataFrame, mode: str):
	"""
		The __add_nodes_mode_claim function creates nodes based on the mode argument.
		The modes are: speaker and year.
			- If the mode is 'speaker', then it will create a node "speaker" for each node in the dataframe with
			relations of "said by" to their respective speakers (Speaker 1 or Speaker 2) as well as add all political
			positions to the speaker's node
			- If the mode is 'year', then it will create a node "year" with the year in which that debate was held and
			it
			will connect to the respective nodes in the dataset with the relation "said in"

		:param df: pd.DataFrame: Pass in the original dataframe
		:param mode: str: Create the nodes based on the mode
		:return: A dataframe with the nodes
	"""
	logger.info(f"Creating nodes with mode: {mode.upper()}")  # based on mode nodes "text + claim/premise"
	if mode == 'speaker':
		governor = df.apply(lambda row: str(row['Governor']) + " " + str(row['G_type']), axis=1)
		dependent = df.apply(lambda row: str(row['Dependent']) + " " + str(row['D_type']), axis=1)
		head = pd.concat([governor, dependent], axis=0)
		tail = pd.concat([df['Speaker1'].copy(), df['Speaker2'].copy()], axis=0)
		assert head.size == tail.size
		speaker_df = pd.DataFrame({'head': head, 'relation': "said by", 'tail': tail})

		role_head = pd.Series(list(political_positions.keys()))
		role_tail = pd.Series(list(political_positions.values()))

		assert role_head.size == role_tail.size
		role_df = pd.DataFrame({'head': role_head, 'relation': "role", 'tail': role_tail})

		final_df = pd.concat([speaker_df, role_df], axis=0)
		final_df = final_df.dropna().reset_index(drop=True)
		return final_df

	elif mode == 'year':
		governor = df.apply(lambda row: str(row['Governor']) + " " + str(row['G_type']), axis=1)
		dependent = df.apply(lambda row: str(row['Dependent']) + " " + str(row['D_type']), axis=1)
		head = pd.concat([governor, dependent], axis=0)
		tail = pd.concat([df['long_date'].copy(), df['long_date'].copy()], axis=0)

		assert head.size == tail.size
		return pd.DataFrame({'head': head, 'relation': "said in", 'tail': tail})


def generate_noise(triplets_file: str, original_triplets_file: str, noisy_triples_file: str, valid_noise: [int]):
	"""
	The generate_noise function takes in a triplets file and generates noisy triples.

	:param triplets_file: str: the path to the file containing all triplets
	:param original_triplets_file: str: the path to the file containing triplets without additional nodes
	:param noisy_triples_file: str: the file path of the noisy triples
	:param valid_noise: [float]: list of the noise ratio
	:return: A dataframe with the following columns: head, relation, tail, noise (1 if it's a synthetic noisy triple,
	0 if not)
	"""
	edges_correct = read_tsv(triplets_file).dropna().reset_index(drop=True)

	# original node distribution
	total_nodes = len(edges_correct)
	for rel in edges_correct['relation'].value_counts().keys():
		percentage = (edges_correct['relation'].value_counts()[rel] / total_nodes) * 100
		logger.info(f"Percentage of {rel}: {percentage}%")

	train, test = train_test_split(edges_correct, test_size=0.2, random_state=123, stratify=edges_correct['relation'])
	train, val = train_test_split(train, test_size=0.15, random_state=123, stratify=train['relation'])

	if config.SPECIAL_BENCHMARKING_FLAG:
		# 	in this case drop from test and val the additional nodes
		original_triples = read_tsv(original_triplets_file).dropna().reset_index(drop=True)
		test = test.merge(original_triples, on=['head', 'relation', 'tail'], how='inner')
		val = val.merge(original_triples, on=['head', 'relation', 'tail'], how='inner')

	train = train.dropna().reset_index(drop=True)
	val = val.dropna().reset_index(drop=True)
	test = test.dropna().reset_index(drop=True)

	for noise_ratio in valid_noise:
		train = __generate_noise(train, noise_ratio=noise_ratio)
		val = __generate_noise(val, noise_ratio=noise_ratio)
		test = __generate_noise(test, noise_ratio=noise_ratio)

		save_tsv(df=train, tsv_file_path=noisy_triples_file.format(noise=noise_ratio, use="train"))
		save_tsv(df=val, tsv_file_path=noisy_triples_file.format(noise=noise_ratio, use="val"))
		save_tsv(df=test, tsv_file_path=noisy_triples_file.format(noise=noise_ratio, use="test"))


def training_strategy(train: pd.DataFrame):
	"""
	Down-sampling strategy

	:param train: DataFrame: the dataframe to balance
	:return: the balanced dataframe
	"""
	logger.info("down-sampling dataset")
	target = len(train[train['relation'] == 'attack'])
	for mode in config.MODE_NODE:
		if mode == 'speaker':
			samples = train[train['relation'] == 'said by'].sample(n=target)
			train.drop(train[train['relation'] == 'said by'].index, inplace=True)
			train = pd.concat([train, samples], axis=0)
			continue
		elif mode == 'year':
			samples = train[train['relation'] == 'said in'].sample(n=target)
			train.drop(train[train['relation'] == 'said in'].index, inplace=True)
			train = pd.concat([train, samples], axis=0)
			continue
		elif mode == 'claim+premise':
			samples = train[train['relation'] == 'is a'].sample(n=target)
			train.drop(train[train['relation'] == 'is a'].index, inplace=True)
			train = pd.concat([train, samples], axis=0)
			continue

	samples = train[train['relation'] == 'support'].sample(n=target)
	train.drop(train[train['relation'] == 'support'].index, inplace=True)
	train = pd.concat([train, samples], axis=0)
	train.dropna().drop_duplicates().reset_index(drop=True)

	return train


def __generate_noise(edges_correct: pd.DataFrame, noise_ratio: float):
	"""
	Helper function that generates noise

	:param edges_correct: str: dataframe containing all the triplets
	:param noise_ratio: float: Determine the amount of noise to be added to the dataset
	:return: A dataframe with the original triplets and noise triplets
	"""
	logger.info(f"generating dataset with {noise_ratio}% noise")

	# ------ special case : leave original dataset as it is ------ #
	if noise_ratio == 0.0:
		edges_correct['noise'] = 0
		return edges_correct

	# ------ special case : half good half fake ------ #
	if noise_ratio == 100:
		noise_to_add = len(edges_correct)
	else:
		noise_to_add = int(math.ceil((noise_ratio / 100) * len(edges_correct)))

	nodes = get_nodes(edges_correct)
	relations = edges_correct['relation'].value_counts().index.to_series().reset_index(drop=True)

	heads = nodes.sample(n=noise_to_add, replace=True).sample(frac=1).reset_index(drop=True)
	tails = nodes.sample(n=noise_to_add, replace=True).sample(frac=1).reset_index(drop=True)
	rel = relations.sample(n=noise_to_add, replace=True).sample(frac=1).reset_index(drop=True)

	edges_noisy = pd.DataFrame({'head': heads, 'relation': rel, 'tail': tails})
	edges_noisy = edges_noisy.dropna().reset_index(drop=True)

	intersection = len(pd.merge(edges_correct, edges_noisy, how='inner'))
	while not intersection == 0:
		# remove duplicates
		edges_noisy = edges_noisy[~edges_noisy.isin(edges_correct.to_dict(orient='list')).all(axis=1)]

		# sample remaining
		heads = nodes.sample(n=intersection, replace=True).sample(frac=1).reset_index(drop=True)
		tails = nodes.sample(n=intersection, replace=True).sample(frac=1).reset_index(drop=True)
		rel = relations.sample(n=intersection, replace=True).sample(frac=1).reset_index(drop=True)

		# concat
		edges_noisy = pd.concat([pd.DataFrame({'head': heads, 'relation': rel, 'tail': tails}),
								 edges_noisy]).dropna().reset_index(drop=True)
		# recompute intersection
		intersection = len(pd.merge(edges_correct, edges_noisy, how='inner'))

	edges_correct['noise'] = 0
	edges_noisy['noise'] = 1  # identify noisy triples

	final_df = pd.concat([edges_correct, edges_noisy], axis=0).sample(frac=1).dropna().reset_index(drop=True)

	return final_df


def get_mapping(dataset: pd.DataFrame):
	"""
	Creates two dictionaries:
		- entity_to_id: maps each entity to an integer id.
		- relation_to_id: maps each relation to an integer id.

	:param dataset: Pd.DataFrame: Get the mapping from entities to ids and relations to ids
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
	Create a TriplesFactory object for each of the train, val and test sets.

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

	# https://github.com/pykeen/pykeen/pull/270
	val_factory = TriplesFactory.from_labeled_triples(triples=val[['head', 'relation', 'tail']].values,
													  entity_to_id=entity_to_id,
													  relation_to_id=relation_to_id,
													  create_inverse_triples=False)

	test_factory = TriplesFactory.from_labeled_triples(triples=test[['head', 'relation', 'tail']].values,
													   entity_to_id=entity_to_id,
													   relation_to_id=relation_to_id,
													   create_inverse_triples=False)

	return train_factory, val_factory, test_factory


def get_factory(dataset: pd.DataFrame, entity_to_id: dict, relation_to_id: dict, create_inverse_triples: bool = False):
	"""
	Creates the TriplesFactory object from the labeled triples of the dataset.

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
	Returns training, testing and validation dataframes

	:param noisy_triples_file: str: The file that contains the noisy triples
	:param noise: float: The amount of noise in the data
	:param drop_col_noise: bool: Drop the noisy column from the dataframe
	:param get_noisy_test: bool: if to return test with noise or not
	:return: A tuple of dataframes, one for each dataset
	"""
	train = read_tsv(noisy_triples_file.format(use="train", noise=noise))

	val = read_tsv(noisy_triples_file.format(use="val", noise=noise))

	if get_noisy_test:
		test = read_tsv(noisy_triples_file.format(use="test", noise=noise))
	else:
		test = read_tsv(noisy_triples_file.format(use="test", noise=0))

	if drop_col_noise:
		train = train.drop("noise", axis=1)
		val = val.drop("noise", axis=1)
		test = test.drop("noise", axis=1)

	return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def generate_mappings(triplets_file: str, triplets_file_utils: str, pretrained_embedding_file: str = None):
	"""
	Creates an entity to id mapping and relation to id mapping

	:param triplets_file: str: the file path of the triplets
	:param triplets_file_utils: str: the path to the directory where to save the entity_to_id and relation_to_id
	mapping
	:param pretrained_embedding_file: file where the pretrained embeddings will be saved
	"""

	logger.info("creating entity-to-id and relation-to-id mappings")
	# read original tsv file
	triplets = read_tsv(triplets_file).dropna().drop_duplicates().reset_index(drop=True)
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

	if config.USE_PRETRAINED_EMBEDDINGS:
		assert pretrained_embedding_file is not None
		logger.info("creating pretrained entity embeddings")
		model = SentenceTransformer('all-MiniLM-L6-v2').to(config.DEVICE)
		embeddings = model.encode(list(entity_to_id.keys()), show_progress_bar=True, device=config.DEVICE)
		numpy.save(pretrained_embedding_file, embeddings)
