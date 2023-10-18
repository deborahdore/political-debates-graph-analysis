import random

import pandas as pd
from loguru import logger
from pykeen.triples import TriplesFactory
from sklearn.model_selection import train_test_split
from utils.utils import load, save


def get_nodes(dataset: pd.DataFrame):
	logger.info("extracting nodes from dataset")

	nodes = pd.concat([dataset['subject'], dataset['object']], axis=0)
	nodes = nodes.dropna().drop_duplicates().reset_index(drop=True)
	nodes = nodes.sample(frac=1) #shuffle
	return nodes


def generate_triplets(original_dataset_file: str, triples_file: str):
	logger.info("generating triplets")
	logger.info(f"original dataset: {original_dataset_file}")
	logger.info(f"destination file: {triples_file}")

	original_dataset = load(original_dataset_file)

	# only useful columns
	df = pd.DataFrame(original_dataset[1:], columns=original_dataset[0])[
		['Dependent', 'D_type', 'Governor', 'G_type', 'RelationType']]

	# create nodes
	df['subject'] = df.apply(lambda row: str(row['Governor']) + "-" + str(row['G_type']), axis=1)
	df['object'] = df.apply(lambda row: str(row['Dependent']) + "-" + str(row['D_type']), axis=1)

	df.drop(columns=['Dependent', 'D_type', 'Governor', 'G_type'], inplace=True)
	df = df[['subject', 'RelationType', 'object']]
	df.columns = ['subject', 'predicate', 'object']

	logger.info("preprocessing created dataset")
	df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
	df = df.dropna().drop_duplicates().reset_index(drop=True)

	columns = df.columns
	save([columns] + df.values.tolist(), triples_file)


def generate_noise(triplets_file: str, noisy_triples_file: str, valid_noise: [float]):
	for noise_ratio in valid_noise:
		df = __generate_noise(triplets_file, noise_ratio=noise_ratio)

		# split
		train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['noisy'])
		train, val = train_test_split(train, test_size=0.15, random_state=42, stratify=train['noisy'])

		columns = df.columns
		save([columns] + train.values.tolist(), noisy_triples_file.format(noise=noise_ratio, use="train"))
		save([columns] + test.values.tolist(), noisy_triples_file.format(noise=noise_ratio, use="test"))
		save([columns] + val.values.tolist(), noisy_triples_file.format(noise=noise_ratio, use="val"))


def __generate_noise(triplets_file: str, noise_ratio: float):
	logger.info(f"generating dataset with {noise_ratio} noise")
	logger.info(f"source file: {triplets_file}")

	edges_correct = load(triplets_file)
	edges_correct = pd.DataFrame(edges_correct[1:], columns=edges_correct[0])

	if noise_ratio == 0.0:
		edges_correct['noisy'] = 0
		return edges_correct

	edges_correct_sample = edges_correct.sample(frac=noise_ratio, random_state=42)
	edges_noisy = []

	nodes = get_nodes(edges_correct)

	for _, (subject, predicate, obj) in edges_correct_sample.iterrows():

		choice = random.random()
		if choice < 0.33:  # remove link from chain
			continue
		elif choice < 0.66:  # swap link
			new_predicate = random.choice(["attack", "support", "equivalent"])
			while new_predicate == predicate:  # avoid same predicate
				new_predicate = random.choice(["attack", "support", "equivalent"])
			edges_noisy.append([subject, new_predicate, obj, 1])

		else:  # introduce new incorrect link
			while True:
				subject_node = random.choice(nodes)
				object_node = random.choice(nodes)
				predicate_label = random.choice(["attack", "support", "equivalent"])

				# Avoid the same node
				if subject_node == object_node:
					continue

				# Check if the combination already exists in edges_correct
				is_combination_unique = (
						(edges_correct['subject'] == subject_node) & (edges_correct['object'] == object_node) & (
						edges_correct['predicate'] == predicate_label))

				if not is_combination_unique.any():
					# The combination is unique, break the loop
					break

			edges_noisy.append([subject_node, predicate_label, object_node, 1])

	noisy_df = pd.DataFrame(edges_noisy, columns=['subject', 'predicate', 'object', 'noisy'])
	noisy_df = noisy_df.dropna().drop_duplicates()

	edges_correct = edges_correct.drop(edges_correct_sample.index)
	edges_correct['noisy'] = 0

	final_df = pd.concat([edges_correct, noisy_df], axis=0)
	final_df = final_df.dropna().drop_duplicates().reset_index(drop=True)

	return final_df


def get_train_val_test_factory(train: pd.DataFrame,
							   val: pd.DataFrame,
							   test: pd.DataFrame,
							   create_inverse_triples: bool = True):
	logger.info("creating train, test and val TriplesFactory")

	train_factory = TriplesFactory.from_labeled_triples(triples=train[['subject', 'predicate', 'object']].values,
														create_inverse_triples=create_inverse_triples)
	val_factory = TriplesFactory.from_labeled_triples(triples=val[['subject', 'predicate', 'object']].values,
													  create_inverse_triples=create_inverse_triples)
	test_factory = TriplesFactory.from_labeled_triples(triples=test[['subject', 'predicate', 'object']].values,
													   create_inverse_triples=create_inverse_triples)

	return train_factory, val_factory, test_factory


def get_factory(dataset: pd.DataFrame, create_inverse_triples: bool = False):
	factory = TriplesFactory.from_labeled_triples(triples=dataset[['subject', 'predicate', 'object']].values,
												  create_inverse_triples=create_inverse_triples)
	return factory


def get_train_val_test_from_dir(noisy_triples_file: str, noise: float, drop_col_noise: bool = True):
	train = load(noisy_triples_file.format(use="train", noise=noise))
	train = pd.DataFrame(data=train[1:], columns=train[0])

	val = load(noisy_triples_file.format(use="val", noise=noise))
	val = pd.DataFrame(data=val[1:], columns=val[0])

	test = load(noisy_triples_file.format(use="test", noise=noise))
	test = pd.DataFrame(data=test[1:], columns=test[0])

	if drop_col_noise:
		train = train.drop("noisy", axis=1)
		val = val.drop("noisy", axis=1)
		test = test.drop("noisy", axis=1)

	return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
