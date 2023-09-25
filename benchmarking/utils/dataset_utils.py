import random

import pandas as pd

from utils.utils import save, load


def get_nodes(dataset: pd.DataFrame) -> pd.DataFrame:
	"""
	The get_nodes function takes a dataset as input and returns a dataframe of all the nodes in that dataset.
	:param dataset: Get the nodes from the dataset
	:return: A dataframe of unique nodes
	"""
	nodes = pd.concat([dataset['node_parent'], dataset['node_child']], axis=0)
	nodes = nodes.dropna().drop_duplicates().reset_index(drop=True)
	return nodes


def generate_triplets(original_dataset_file: str, dataset_file: str) -> None:
	"""
	The generate_triplets function takes in two arguments:
		- original_dataset_file: the path to the original dataset file
		- dataset_file: the path to where you want to save your new triplet file

	:param original_dataset_file: str: Specify the path to the original dataset file
	:param dataset_file: str: Specify the file where the dataset will be stored
	"""
	original_dataset = load(original_dataset_file)

	# only useful columns
	df = pd.DataFrame(original_dataset[1:], columns=original_dataset[0])[
		['Dependent', 'D_type', 'Governor', 'G_type', 'RelationType']]
	df['node_parent'] = df.apply(lambda row: str(row['Governor']) + "-" + str(row['G_type']), axis=1)
	df['node_child'] = df.apply(lambda row: str(row['Dependent']) + "-" + str(row['D_type']), axis=1)
	df.drop(columns=['Dependent', 'D_type', 'Governor', 'G_type'], inplace=True)
	df = df[['node_parent', 'RelationType', 'node_child']]
	df.columns = ['node_parent', 'relation', 'node_child']

	# lowercase
	df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
	df = df.dropna().drop_duplicates().reset_index(drop=True)

	save([df.columns] + df.values.tolist(), dataset_file)


def generate_noise(dataset_file: str, noisy_dataset_file: str, noise_ratio: float) -> pd.DataFrame:
	"""
	The generate_noise function takes a dataset file, a noisy dataset file and the noise ratio as input.
	It returns the dataset with noisy edges in form of a pandas dataframe. Each noisy edge has 1 as label
	will label 0 indicates a correct edge.
	Possible operations to introduce noise are: remove links, swap link label, introduce new incorrect links

	:param dataset_file: str: Specify the dataset file to be used
	:param noisy_dataset_file: str: Specify the name of the file where noisy dataset will be stores
	:param noise_ratio: float: Determine the amount of noise to be added to the dataset
	:return: A dataframe of the noisy edges
	"""

	edges_correct = load(dataset_file)
	edges_correct = pd.DataFrame(edges_correct[1:], columns=edges_correct[0])

	edges_correct_sample = edges_correct.sample(frac=noise_ratio, random_state=42)
	edges_noisy = []

	nodes = get_nodes(edges_correct)


	for _, (node_parent, relation, node_child) in edges_correct_sample.iterrows():

		choice = random.random()
		if choice < 0.33:  # remove link from chain
			continue
		elif choice < 0.66:  # swap link
			new_relation = random.choice(["attack", "support", "equivalent"])
			while new_relation == relation: # avoid same relation
				new_relation = random.choice(["attack", "support", "equivalent"])
			edges_noisy.append([node_parent, new_relation, node_child, 1])

		else:  # introduce new incorrect link
			while True:
				source_node = random.choice(nodes)
				target_node = random.choice(nodes)
				edge_label = random.choice(["attack", "support", "equivalent"])

				# Avoid the same node
				if source_node == target_node:
					continue

				# Check if the combination already exists in edges_correct
				is_combination_unique = ((edges_correct['node_parent'] == source_node) &
										 (edges_correct['node_child'] == target_node) &
										 (edges_correct['relation'] == edge_label))

				if not is_combination_unique.any():
					# The combination is unique, break the loop
					break

			edges_noisy.append([source_node, edge_label, target_node, 1])

	noisy_df = pd.DataFrame(edges_noisy, columns=['node_parent', 'relation', 'node_child', 'noisy'])
	noisy_df = noisy_df.dropna().drop_duplicates()

	edges_correct = edges_correct.drop(edges_correct_sample.index)
	edges_correct['noisy'] = 0

	final_df = pd.concat([edges_correct, noisy_df], axis=0)
	final_df = final_df.sample(frac=1).dropna().drop_duplicates().reset_index(drop=True)
	save([final_df.columns] + final_df.values.tolist(), noisy_dataset_file.format(ratio=noise_ratio))

	return final_df
