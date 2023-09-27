import ast
import sys

from benchmark import do_benchmarking
from config.config import dataset_file, model_dir, noisy_dataset_file, original_dataset_file, pipeline_config_file, \
	plot_file, ratio, metric_file, model_file, valid_kge_models
from utils.dataset_utils import generate_noise, generate_triplets
from loguru import logger

if __name__ == '__main__':
	logger.info("start")

	if len(sys.argv) < 2:
		logger.error("missing arguments from command line")
		raise Exception("missing arguments from command line")

	generate_dataset = ast.literal_eval(sys.argv[1])

	if generate_dataset:
		
		generate_triplets(original_dataset_file, dataset_file)
		for r in ratio:
			generate_noise(dataset_file=dataset_file, noisy_dataset_file=noisy_dataset_file, noise_ratio=r)

	for model in valid_kge_models:
		for r in ratio:
			do_benchmarking(model,
							noisy_dataset_file,
							model_file,
							pipeline_config_file,
							plot_file,
							metric_file,
							model_dir,
							r)

	logger.info("end")
