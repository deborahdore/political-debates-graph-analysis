import csv


def load(file: str) -> []:
	"""
	The load function reads the dataset from a csv file and returns it as a list of lists.
	:return: A list of lists
	"""
	dataset = []
	with open(file, 'r') as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			dataset.append(row)

	return dataset


def save(dataset: [], csv_filename: str) -> None:
	"""
	The save function takes a dataset and saves it to a csv file.

	:param dataset: []: Pass in the dataset to be saved
	:param csv_filename: str: Specify the file name of the csv file to be saved
	"""
	with open(csv_filename, 'w', newline='') as csvfile:
		csv_writer = csv.writer(csvfile)
		csv_writer.writerows(dataset)
