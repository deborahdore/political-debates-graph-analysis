import os.path
from pathlib import Path

import pandas as pd

from config.config import excel_dir, metrics_file, results_dir, valid_kge_models, valid_noise_ratio
from utils.utils import read_json, save_json

if __name__ == '__main__':

	tasks = ['link prediction', 'link deletion', 'triple classification']

	link_prediction_data = {}
	link_deletion_data = {}
	triple_classification_data = {}

	for model in valid_kge_models:
		for noise in valid_noise_ratio:
			metrics = read_json(metrics_file.format(model=model, ratio=noise))
			link_prediction_data[noise][model] = metrics['link prediction']
			link_deletion_data[noise][model] = metrics['link deletion']
			triple_classification_data[noise][model] = metrics['triple classification']

	save_json(link_prediction_data, os.path.join(results_dir, "link_prediction.json"))
	save_json(link_deletion_data, os.path.join(results_dir, "link_deletion.json"))
	save_json(triple_classification_data, os.path.join(results_dir, "triple_classification.json"))

	# create excel
	for noise in link_prediction_data.keys():
		file = os.path.join(excel_dir, f"link_prediction_{noise}.xlsx")
		pd.DataFrame(link_prediction_data[noise]).to_excel(file)

	for noise in link_deletion_data.keys():
		file = os.path.join(excel_dir, f"link_deletion_{noise}.xlsx")
		pd.DataFrame(link_deletion_data[noise]).to_excel(file)

	for noise in link_prediction_data.keys():
		file = os.path.join(excel_dir, f"triple_classification_{noise}.xlsx")
		pd.DataFrame(triple_classification_data[noise]).to_excel(file)

