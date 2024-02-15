import json
import os

import pandas as pd

""" Place this file inside the directory basic/special/pretrained/ where you want to extract data from """

### COLUMN NAMES ###
link_prediction = 'Link Prediction HITS@10'
link_deletion = 'Link Deletion HITS@10'
relation_prediction = 'Relation Prediction HITS@1'
triple_classification = 'Triple Classification F1-Macro'
relation_classification = 'Relation Classification F1-Macro'


def extract_data_from_json(json_path):
	with open(json_path, 'r') as f:
		data = json.load(f)
	return data


def process_directory(directory):
	data = {}
	for model_folder in ["TransE", "DistMult", "ConvE"]:
		model_path = os.path.join(directory, model_folder)
		data[model_folder] = {}
		result_path = os.path.join(model_path, 'results')
		data[model_folder][link_prediction] = \
			extract_data_from_json(os.path.join(result_path, "link_prediction_noise_0.json"))['both']['hits_at_10']

		data[model_folder][link_deletion] = \
			extract_data_from_json(os.path.join(result_path, "link_deletion_noise_0.json"))['both']['hits_at_10']

		data[model_folder][relation_prediction] = \
			extract_data_from_json(os.path.join(result_path, "relation_prediction_noise_0.json"))['hits_at_1']

		data[model_folder][triple_classification] = \
			extract_data_from_json(os.path.join(result_path, "triple_classification_noise_0.json"))['f1_macro']

		data[model_folder][relation_classification] = \
			extract_data_from_json(os.path.join(result_path, "relation_classification_noise_0.json"))['f1_macro']

	return data


def main():
	current_directory = os.getcwd()
	task = current_directory.split('/')[-1]
	data = {}
	for folder in os.listdir(current_directory):  # for every folder in basic/special/pretrained
		folder_path = os.path.join(current_directory, folder)
		data[folder] = process_directory(folder_path)

	df_rows = []
	for folder, models in data.items():
		for model, metrics in models.items():
			row = {
				'Experiment'           : f"{task}/{folder}",
				'Model'                : model,
				link_prediction        : metrics.get(link_prediction, None),
				link_deletion          : metrics.get(link_deletion, None),
				relation_prediction    : metrics.get(relation_prediction, None),
				triple_classification  : metrics.get(triple_classification, None),
				relation_classification: metrics.get(relation_classification, None)}
			df_rows.append(row)

	df = pd.DataFrame(df_rows)
	df.to_excel(f"extracted_data_{task}.xlsx", index=False)


if __name__ == "__main__":
	main()
