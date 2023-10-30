import os.path

import pandas as pd
from config.config import metrics_file, results_dir, valid_models, valid_noise_ratio
from utils.utils import read_json, save_json


def highlight_max(s):
	is_max = s == s.max()
	return ['font-weight: bold' if v else '' for v in is_max]


def process_results():
	link_prediction_data = {}
	link_deletion_data = {}
	triple_classification_data = {}

	for noise in valid_noise_ratio:
		link_prediction_data_noise = {}
		link_deletion_data_noise = {}
		triple_classification_data_noise = {}

		for model in valid_models:
			metrics = read_json(metrics_file.format(model=model, ratio=noise))
			link_prediction_data_noise.update({model: metrics['link prediction']['both']})
			link_deletion_data_noise.update({model: metrics['link deletion']['both']})
			triple_classification_data_noise.update({model: metrics['triple classification']})

		link_prediction_data.update({str(noise): link_prediction_data_noise})
		link_deletion_data.update({str(noise): link_deletion_data_noise})
		triple_classification_data.update({str(noise): triple_classification_data_noise})

	save_json(link_prediction_data, os.path.join(results_dir, "link_prediction.json"))
	save_json(link_deletion_data, os.path.join(results_dir, "link_deletion.json"))
	save_json(triple_classification_data, os.path.join(results_dir, "triple_classification.json"))

	with pd.ExcelWriter(os.path.join(results_dir, 'excel/link_prediction.xlsx')) as writer:
		for noise in valid_noise_ratio:
			df = pd.DataFrame(link_prediction_data[str(noise)])
			df = df.style.apply(highlight_max, axis=1)
			df.to_excel(writer, sheet_name=str(noise), float_format="%.5f")

	with pd.ExcelWriter(os.path.join(results_dir, 'excel/link_deletion.xlsx')) as writer:
		for noise in valid_noise_ratio:
			df = pd.DataFrame(link_deletion_data[str(noise)])
			df = df.style.apply(highlight_max, axis=1)
			df.to_excel(writer, sheet_name=str(noise), float_format="%.5f", engine='openpyxl', index=False)

	with pd.ExcelWriter(os.path.join(results_dir, 'excel/triple_classification.xlsx')) as writer:
		for noise in valid_noise_ratio:
			df = pd.DataFrame(triple_classification_data[str(noise)])
			df = df.style.apply(highlight_max, axis=1)
			df.to_excel(writer, sheet_name=str(noise), float_format="%.5f")
