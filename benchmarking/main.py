import ast
import os.path
import sys

from config import config
from evaluation import link_deletion, \
	link_deletion_bert, \
	link_prediction, \
	link_prediction_bert, \
	relation_classification, \
	triple_classification, \
	triple_classification_bert
from training import bert_training, hyperparameter_optimization, training
from utils.dataset_utils import generate_mappings, generate_noise, generate_triplets
from utils.utils import load_model


def get_kwargs():
	"""
	The get_kwargs function parses the command line arguments and returns them as a tuple.
	The function first checks that all the required arguments are present in sys.argv, then it
	parses each argument into its respective type (e.g., int, float, str). The function returns a
	tuple containing these parsed values.

	:return: A tuple of four elements:
	- generate argument containing True/False. If true, the datasets with noise will be generated
	- optimize argument containing True/False. If true, will perform hyperparameter optimization
	- model argument containing the model name to be optimized/trained
	- noise argument containing the noise ratio to optimize/train the model on

	"""
	assert "--generate" in sys.argv
	assert "--optimize" in sys.argv
	assert "--model" in sys.argv
	assert "--noise" in sys.argv

	gen_arg = ast.literal_eval(sys.argv[sys.argv.index("--generate") + 1])
	opt_arg = ast.literal_eval(sys.argv[sys.argv.index("--optimize") + 1])
	noise_arg = int(sys.argv[sys.argv.index("--noise") + 1])
	model_arg = sys.argv[sys.argv.index("--model") + 1]

	assert noise_arg in config.VALID_NOISE_RATIO
	assert model_arg in config.VALID_MODELS

	return gen_arg, opt_arg, noise_arg, model_arg


def bert_basic(model_name: str, noise: int):
	"""
	The bert_basic function is used to train a BERT model on the noisy triples file, and then evaluate it using link
	prediction, link deletion, and triple classification.

	:param model_name:str: Name the model
	:param noise:int: Specify the noise ratio
	"""
	model_file = os.path.join(config.model_dir.format(model="Bert"), f"{noise}/{model_name}_{noise}.pt")
	if not os.path.exists(model_file) or config.FORCE_TRAINING:
		model = bert_training(model_file=model_file,
							  model_name='Bert',
							  noisy_triples_file=config.noisy_triples_file,
							  ratio=noise)
	else:
		model = load_model(model_file)

	link_prediction_bert(model=model,
						 model_dir=config.model_dir.format(model="Bert"),
						 model_name='Bert',
						 noisy_triples_file=config.noisy_triples_file,
						 metrics_file=config.metrics_file.format(model=model_name, ratio=noise),
						 noise_ratio=noise)
	link_deletion_bert(model=model,
					   model_dir=config.model_dir.format(model="Bert"),
					   model_name='Bert',
					   noisy_triples_file=config.noisy_triples_file,
					   metrics_file=config.metrics_file.format(model=model_name, ratio=noise),
					   noise_ratio=noise)
	triple_classification_bert(model=model,
							   model_dir=config.model_dir.format(model="Bert"),
							   model_name='Bert',
							   noisy_triples_file=config.noisy_triples_file,
							   metrics_file=config.metrics_file.format(model=model_name, ratio=noise),
							   noise_ratio=noise)


def kge_basic(model_name: str, noise: int):
	"""
	The kge_basic function is a wrapper function that trains and evaluates the KGE model.

	:param model_name: str: Name the model
	:param noise: int: Specify the noise ratio of the dataset
	"""
	model_file = os.path.join(config.model_dir.format(model=model_name), f"{noise}/{model_name}_{noise}.pt")
	if not os.path.exists(model_file) or config.FORCE_TRAINING:
		result = training(model_dir=config.model_dir.format(model=model_name),
						  model_name=model_name,
						  model_file=model_file,
						  noisy_triples_file=config.noisy_triples_file,
						  triplets_file_utils=config.triplets_file_utils,
						  ratio=noise,
						  pretrained_embedding_file=config.pretrained_embedding_file)
		model = result.model

	else:
		model = load_model(model_file)

	link_prediction(model=model,
					noisy_triples_file=config.noisy_triples_file,
					triplets_file_utils=config.triplets_file_utils,
					model_name=model_name,
					metrics_file=config.metrics_file.format(model=model_name, ratio=noise),
					noise_ratio=noise)
	link_deletion(model=model,
				  model_name=model_name,
				  noisy_triples_file=config.noisy_triples_file,
				  triplets_file_utils=config.triplets_file_utils,
				  metrics_file=config.metrics_file.format(model=model_name, ratio=noise),
				  noise_ratio=noise)
	triple_classification(model=model,
						  model_name=model_name,
						  noisy_triples_file=config.noisy_triples_file,
						  triplets_file_utils=config.triplets_file_utils,
						  metrics_file=config.metrics_file.format(model=model_name, ratio=noise),
						  noise_ratio=noise)

	relation_classification(model=model,
							model_name=model_name,
							noisy_triples_file=config.noisy_triples_file,
							triplets_file_utils=config.triplets_file_utils,
							metrics_file=config.metrics_file.format(model=model_name, ratio=noise),
							noise_ratio=noise)


if __name__ == '__main__':
	generate_dataset, optimization, noise, model_name = get_kwargs()

	if generate_dataset:
		# generates triples from original file
		generate_triplets(original_dataset_file=config.original_dataset_file,
						  triples_file=config.triplets_file,
						  original_triplets_file=config.original_triplets_file)
		# generate node to label mappings
		generate_mappings(triplets_file=config.triplets_file,
						  triplets_file_utils=config.triplets_file_utils,
						  pretrained_embedding_file=config.pretrained_embedding_file)
		# for every level of noise, create a dataset
		generate_noise(triplets_file=config.triplets_file,
					   original_triplets_file=config.original_triplets_file,
					   noisy_triples_file=config.noisy_triples_file,
					   valid_noise=config.VALID_NOISE_RATIO)
	if optimization:
		if model_name == 'Bert': exit(1)
		hyperparameter_optimization(model_name=model_name,
									model_dir=config.model_dir.format(model=model_name),
									noisy_triples_file=config.noisy_triples_file,
									triplets_file_utils=config.triplets_file_utils,
									pretrained_embedding_file=config.pretrained_embedding_file)

	else:
		if model_name == 'Bert':
			bert_basic(model_name, noise)
		else:
			kge_basic(model_name, noise)
