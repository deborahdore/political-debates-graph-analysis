import math

import numpy
import pandas as pd
from loguru import logger
from pykeen.triples import TriplesFactory
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

from benchmarking import config
from benchmarking.utils.util import read_csv, read_json, read_tsv, save_json, save_tsv


def get_nodes(dataset: pd.DataFrame):
	""" Finds the unique nodes in the dataset """
	nodes = pd.concat([dataset['head'], dataset['tail']], axis=0)
	nodes = nodes.sample(frac=1)  # shuffle
	nodes = nodes.dropna().drop_duplicates().reset_index(drop=True)
	return nodes


def generate_triplets(original_dataset_file: str,
					  original_split_triplets_file: str,
					  noisy_split_triplets_file: str,
					  noisy_triplets_file: str):
	""" Generate additional triples from original dataset """

	noisy_triplets_file = noisy_triplets_file.format(noise=0)

	logger.info("💡 Triplets generation")

	# load original dataset
	df_original = read_csv(original_dataset_file)
	df_original['RelationType'] = df_original['RelationType'].replace({
		'support'   : '__label__support',
		'attack'    : '__label__attack',
		'equivalent': '__label__equivalent'})

	train_df = read_tsv(original_split_triplets_file.format(split="train"))
	dev_df = read_tsv(original_split_triplets_file.format(split="dev"))
	test_df = read_tsv(original_split_triplets_file.format(split="test"))

	if config.MODE_TEXT == 'text+type':
		train_df = configure_mode_text_type(df_original, train_df)
		dev_df = configure_mode_text_type(df_original, dev_df)
		test_df = configure_mode_text_type(df_original, test_df)

	if not bool(config.MODE_NODE):
		for mode_node in config.MODE_NODE:
			mode_df = extract_nodes(df_original, mode_node, config.MODE_TEXT)

			train, test = train_test_split(mode_df,
										   test_size=0.2,
										   random_state=config.SEED,
										   stratify=mode_df['relation'])
			test, val = train_test_split(test, test_size=0.5, random_state=config.SEED, stratify=test['relation'])

			train_df = pd.concat([train_df, train], axis=0)
			dev_df = pd.concat([dev_df, val], axis=0)
			test_df = pd.concat([test_df, test], axis=0)

	save_tsv(train_df, noisy_split_triplets_file.format(noise=0, split="train"))
	save_tsv(test_df, noisy_split_triplets_file.format(noise=0, split="test"))
	save_tsv(dev_df, noisy_split_triplets_file.format(noise=0, split="dev"))

	final_df = pd.concat([train_df, dev_df], axis=0)
	final_df = pd.concat([final_df, test_df], axis=0)
	save_tsv(final_df, noisy_triplets_file.format(noise=0))


def configure_mode_text_type(df_original: pd.DataFrame, df: pd.DataFrame):
	""" Transform nodes from only text to text+type"""

	# extract claims and premises from dataframe
	claim_premises_df = pd.DataFrame()
	claim_premises_df['head'] = df_original['Governor']
	claim_premises_df['type_head'] = df_original['Governor'] + '_' + df_original['G_type']
	claim_premises_df['tail'] = df_original['Dependent']
	claim_premises_df['type_tail'] = df_original['Dependent'] + '_' + df_original['D_type']
	claim_premises_df['relation'] = df_original['RelationType']

	# respect original split
	intersection = pd.merge(claim_premises_df, df, on=['head', 'relation', 'tail'], how='inner')
	return pd.DataFrame({
		'relation': intersection['relation'],
		'head'    : intersection['type_head'],
		'tail'    : intersection['type_tail']}).dropna().drop_duplicates().reset_index(drop=True)


def extract_nodes(df_original: pd.DataFrame, mode_node: str, mode_text: str):
	""" Extract nodes based on mode """
	df = pd.DataFrame()
	if mode_text == 'text':
		if mode_node == 'claim+premise':
			df['head'] = pd.concat([df_original['Dependent'], df_original['Governor']], axis=0)
			df['tail'] = pd.concat([df_original['D_type'], df_original['G_type']], axis=0)
			df['relation'] = "__label__type"
		elif mode_node == 'year':
			df['head'] = pd.concat([df_original['Dependent'], df_original['Governor']], axis=0)
			df['tail'] = pd.concat([df_original['long_date'], df_original['long_date']], axis=0)
			df['relation'] = "__label__year"
		elif mode_node == 'speaker':
			df['head'] = pd.concat([df_original['Dependent'], df_original['Governor']], axis=0)
			df['tail'] = pd.concat([df_original['Speaker1'], df_original['Speaker2']], axis=0)
			df['relation'] = "__label__says"
	elif mode_text == 'text+claim':
		if mode_node == 'year':
			head1 = df_original['Governor'] + '_' + df_original['G_type']
			head2 = df_original['Dependent'] + '_' + df_original['D_type']
			df['head'] = pd.concat([head1, head2], axis=0)
			df['tail'] = pd.concat([df_original['long_date'], df_original['long_date']], axis=0)
			df['relation'] = "__label__year"
		elif mode_node == 'speaker':
			head1 = df_original['Dependent'] + '_' + df_original['D_type']
			head2 = df_original['Governor'] + '_' + df_original['G_type']
			df['head'] = pd.concat([head1, head2], axis=0)
			df['tail'] = pd.concat([df_original['Speaker1'], df_original['Speaker2']], axis=0)
			df['relation'] = "__label__says"

	assert not df.empty
	return df.dropna().drop_duplicates().reset_index(drop=True)


def generate_noise(noisy_triplets_file: str, noisy_split_triplets_file: str, valid_noise: [int]):
	""" For every possible noise value, generate a new dataset	"""

	correct = read_tsv(noisy_triplets_file.format(noise=0))

	# original node distribution
	total_nodes = len(correct)
	for rel in correct['relation'].value_counts().keys():
		percentage = (correct['relation'].value_counts()[rel] / total_nodes) * 100
		logger.info(f"📊 Percentage of {rel}: {percentage}%")

	train = read_tsv(noisy_split_triplets_file.format(noise=0, split='train'))[['head', 'relation', 'tail']]
	train['noise'] = 0
	dev = read_tsv(noisy_split_triplets_file.format(noise=0, split='dev'))[['head', 'relation', 'tail']]
	dev['noise'] = 0
	test = read_tsv(noisy_split_triplets_file.format(noise=0, split='test'))[['head', 'relation', 'tail']]
	test['noise'] = 0

	for noise_ratio in valid_noise:
		if noise_ratio == 0:
			save_tsv(train, noisy_split_triplets_file.format(noise=noise_ratio, split="train"))
			save_tsv(test, noisy_split_triplets_file.format(noise=noise_ratio, split="test"))
			save_tsv(dev, noisy_split_triplets_file.format(noise=noise_ratio, split="dev"))
			continue

		# ------ special case : half good half fake ------ #
		if noise_ratio == 100:
			noise_to_add = len(correct)
		else:
			noise_to_add = int(math.ceil((noise_ratio / 100) * len(correct)))

		nodes = get_nodes(correct)
		relations = correct['relation'].value_counts().index.to_series().reset_index(drop=True)
		heads = nodes.sample(n=noise_to_add, replace=True).sample(frac=1).reset_index(drop=True)
		tails = nodes.sample(n=noise_to_add, replace=True).sample(frac=1).reset_index(drop=True)
		rel = relations.sample(n=noise_to_add, replace=True).sample(frac=1).reset_index(drop=True)

		noisy = pd.DataFrame({'head': heads, 'relation': rel, 'tail': tails}).dropna().reset_index(drop=True)
		intersection = len(pd.merge(correct, noisy, how='inner'))

		while not intersection == 0:
			# remove duplicates
			noisy = noisy[~noisy.isin(correct.to_dict(orient='list')).all(axis=1)]

			# sample remaining
			heads = nodes.sample(n=intersection, replace=True).sample(frac=1).reset_index(drop=True)
			tails = nodes.sample(n=intersection, replace=True).sample(frac=1).reset_index(drop=True)
			rel = relations.sample(n=intersection, replace=True).sample(frac=1).reset_index(drop=True)

			# concat
			noisy = pd.concat([pd.DataFrame({'head': heads, 'relation': rel, 'tail': tails}),
							   noisy]).dropna().reset_index(drop=True)
			# recompute intersection
			intersection = len(pd.merge(correct, noisy, how='inner'))

		train_noisy, test_noisy = train_test_split(noisy,
												   test_size=0.2,
												   random_state=config.SEED,
												   stratify=noisy['relation'])
		test_noisy, dev_noisy = train_test_split(test_noisy,
												 test_size=0.5,
												 random_state=config.SEED,
												 stratify=test_noisy['relation'])

		train_noisy['noise'] = 1
		test_noisy['noise'] = 1
		dev_noisy['noise'] = 1

		new_train_noisy = pd.concat([train, train_noisy], axis=0)
		new_test_noisy = pd.concat([test_noisy, test], axis=0)
		new_dev_noisy = pd.concat([dev_noisy, dev], axis=0)

		save_tsv(new_train_noisy, noisy_split_triplets_file.format(noise=noise_ratio, split="train"))
		save_tsv(new_test_noisy, noisy_split_triplets_file.format(noise=noise_ratio, split="test"))
		save_tsv(new_dev_noisy, noisy_split_triplets_file.format(noise=noise_ratio, split="dev"))


def get_mapping(dataset: pd.DataFrame):
	""" Get the mapping from entities to ids and relations to ids from the dataset """
	dataset = TriplesFactory.from_labeled_triples(triples=dataset[['head', 'relation', 'tail']].values,
												  create_inverse_triples=False)
	return dataset.entity_to_id, dataset.relation_to_id


def get_train_val_test_factory(train: pd.DataFrame,
							   val: pd.DataFrame,
							   test: pd.DataFrame,
							   triplets_file_utils: str,
							   create_inverse_triples: bool):
	"""	Create a TriplesFactory object for each of the train, val and test sets. """
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
	""" Creates the TriplesFactory object from the labeled triples of the dataset. """
	factory = TriplesFactory.from_labeled_triples(triples=dataset[['head', 'relation', 'tail']].values,
												  entity_to_id=entity_to_id,
												  relation_to_id=relation_to_id,
												  create_inverse_triples=create_inverse_triples)
	return factory


def get_train_val_test_from_dir(noisy_triples_file: str,
								noise: float,
								drop_col_noise: bool = True,
								get_noisy_test: bool = False,
								special_benchmarking_flag: bool = config.SPECIAL_BENCHMARKING_FLAG):
	"""	Returns training, testing and validation dataframes	"""

	train = read_tsv(noisy_triples_file.format(split="train", noise=noise))
	val = read_tsv(noisy_triples_file.format(split="dev", noise=noise))

	if get_noisy_test:
		test = read_tsv(noisy_triples_file.format(split="test", noise=noise))
	else:
		test = read_tsv(noisy_triples_file.format(split="test", noise=0))

	if drop_col_noise:
		train = train.drop("noise", axis=1)
		val = val.drop("noise", axis=1)
		test = test.drop("noise", axis=1)

	if special_benchmarking_flag:
		val = val[(val['relation'] == '__label__support') or (val['relation'] == '__label__attack') or (
				val['relation'] == '__label__equivalent')]

		test = test[(test['relation'] == '__label__support') or (test['relation'] == '__label__attack') or (
				test['relation'] == '__label__equivalent')]

	return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def generate_mappings(noisy_triplets_file: str, triplets_file_utils: str, pretrained_embedding_file: str = None):
	"""	Creates an entity to id mapping and relation to id mapping	"""
	triplets = read_tsv(noisy_triplets_file.format(noise=0))

	factory = TriplesFactory.from_labeled_triples(triples=triplets[['head', 'relation', 'tail']].values,
												  create_inverse_triples=False)

	entity_to_id = factory.entity_to_id
	entity_to_id['placeholder'] = len(entity_to_id.keys())
	relation_to_id = factory.relation_to_id

	entity_to_id_file = triplets_file_utils.format(file_name="entity_to_id")
	save_json(entity_to_id, entity_to_id_file)

	relation_to_id_file = triplets_file_utils.format(file_name="relation_to_id")
	save_json(relation_to_id, relation_to_id_file)

	if config.USE_PRETRAINED_EMBEDDINGS:
		assert pretrained_embedding_file is not None
		model = SentenceTransformer('all-MiniLM-L6-v2').to(config.DEVICE)
		embeddings = model.encode(list(entity_to_id.keys()), show_progress_bar=True, device=config.DEVICE)
		numpy.save(pretrained_embedding_file, embeddings)
