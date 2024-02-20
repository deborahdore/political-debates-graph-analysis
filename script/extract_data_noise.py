import json
import os

import pandas as pd

""" This file is used to extract the data of a SINGLE TASK eg. text+type-year for multiple levels of NOISE"""
""" Place it inside the directory where you want to extract the data """
""" eg. /your/path/to/output/pretrained/text+type-year/extract_data_noise.py """

### NOISE LEVEL TO EXTRACT ###
NOISE = [0]

### COLUMN NAMES ###
link_prediction = 'Link Prediction HITS@10'
link_deletion = 'Link Deletion HITS@10'
relation_prediction_hits1 = 'Relation Prediction HITS@1'
relation_prediction_hits2 = 'Relation Prediction HITS@2'
triple_classification = 'Triple Classification F1-Macro'
relation_classification = 'Relation Classification F1-Macro'
predictions = 'Relation Classification + Prediction F1-Macro'
support_prediction = 'Support Prediction HITS@1'
attack_prediction = 'Attack Prediction HITS@1'
equivalent_prediction = 'Equivalent Prediction HITS@1'


def extract_data_from_json(json_path):
	with open(json_path, 'r') as f:
		data = json.load(f)
	return data


def process_directory(folder):
	data = {}
	for noise in NOISE:
		result_path = os.path.join(folder, f"results/noise_{noise}")
		print(result_path)
		data[noise] = {}

		link_prediction_file = os.path.join(result_path, "link_prediction.json")
		assert os.path.isfile(link_prediction_file)

		link_deletion_file = os.path.join(result_path, "link_deletion.json")
		assert os.path.isfile(link_deletion_file)

		relation_prediction_file = os.path.join(result_path, "relation_prediction.json")
		assert os.path.isfile(relation_prediction_file)

		triple_classification_file = os.path.join(result_path, "triple_classification.json")
		assert os.path.isfile(triple_classification_file)

		relation_classification_file = os.path.join(result_path, "relation_classification.json")
		assert os.path.isfile(relation_classification_file)

		predictions_eval_file = os.path.join(result_path, "predictions_eval.json")
		assert os.path.isfile(predictions_eval_file)

		support_prediction_file = os.path.join(result_path, "__label__Support_prediction.json")
		assert os.path.isfile(support_prediction_file)

		attack_prediction_file = os.path.join(result_path, "__label__Attack_prediction.json")
		assert os.path.isfile(attack_prediction_file)

		equivalent_prediction_file = os.path.join(result_path, "__label__Equivalent_prediction.json")
		assert os.path.isfile(equivalent_prediction_file)

		data[noise][link_prediction] = extract_data_from_json(link_prediction_file)['both']['hits_at_10']
		data[noise][link_deletion] = extract_data_from_json(link_deletion_file)['both']['hits_at_10']

		data[noise][relation_prediction_hits1] = extract_data_from_json(relation_prediction_file)['hits_at_1']
		data[noise][relation_prediction_hits2] = extract_data_from_json(relation_prediction_file)['hits_at_2']

		data[noise][triple_classification] = extract_data_from_json(triple_classification_file)['f1_macro']
		data[noise][relation_classification] = extract_data_from_json(relation_classification_file)['f1_macro']

		data[noise][predictions] = extract_data_from_json(predictions_eval_file)['f1_macro']

		data[noise][support_prediction] = extract_data_from_json(support_prediction_file)['hits_at_1']
		data[noise][attack_prediction] = extract_data_from_json(attack_prediction_file)['hits_at_1']
		data[noise][equivalent_prediction] = extract_data_from_json(equivalent_prediction_file)['hits_at_1']

	return data


def main():
	current_directory = os.getcwd()
	task = current_directory.split('/')[-1]
	data = {}
	counter = 0
	for folder in os.listdir(current_directory):
		folder_path = os.path.join(current_directory, folder)
		if not os.path.isdir(folder_path):
			continue
		counter += 1
		data[folder] = process_directory(folder_path)

	df_rows = []
	for model, results in data.items():
		for noise, metrics in results.items():
			row = {
				f'Experiment'            : task,
				'Noise'                  : noise,
				'Model'                  : model,
				link_prediction          : metrics.get(link_prediction, None),
				link_deletion            : metrics.get(link_deletion, None),
				triple_classification    : metrics.get(triple_classification, None),
				relation_prediction_hits1: metrics.get(relation_prediction_hits1, None),
				relation_prediction_hits2: metrics.get(relation_prediction_hits2, None),
				support_prediction       : metrics.get(support_prediction, None),
				attack_prediction        : metrics.get(attack_prediction, None),
				equivalent_prediction    : metrics.get(equivalent_prediction, None),
				relation_classification  : metrics.get(relation_classification, None),
				predictions              : metrics.get(predictions, None)}
			df_rows.append(row)

	df = pd.DataFrame(df_rows)
	assert not df.empty
	df.to_excel(f"extracted_{task}.xlsx", index=False)


if __name__ == "__main__":
	main()
