import pandas as pd

from benchmarking.utils.utils import load, save


def generate_triplets(original_dataset_file: str, dataset_file: str) -> None:
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
	df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
	df = df.dropna().drop_duplicates().reset_index(drop=True)

	save([df.columns.tolist()] + df.values.tolist(), dataset_file)
