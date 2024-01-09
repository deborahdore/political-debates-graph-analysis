import numpy
import pandas as pd
import torch
from loguru import logger
from pykeen.triples import TriplesFactory
from sentence_transformers import SentenceTransformer

from utils.utils import save_json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = "/dataset/{use}_relations_new.tsv"
triplets_file_utils = "/dataset/{file_name}.json"
pretrained_embedding_file = "/dataset/pretrained_embedding_file.npy"


def generate_mappings(triplets_file: str, triplets_file_utils: str, pretrained_embedding_file: str = None):
	"""
	The generate_mappings function takes in a triplets file and a triplets_file_utils string.
	The function reads the triplet file, creates an entity to id mapping and relation to id mapping,
	and saves them as json files into the triplets_file_utils file.

	:param triplets_file: str: Specify the file path of the triplets
	:param triplets_file_utils: str: Specify the path to the directory where you want to save your entity_to_id and
	relation_to_id mapping
	:param pretrained: whether to use bert for the embeddings
	:param pretrained_embedding_file: where to save pretrained embeddings
	"""

	logger.info("creating entity-to-id and relation-to-id mappings")
	# read original tsv file
	triplets = pd.read_csv(triplets_file.format(use="full"),
						   index_col=False,
						   header=0).dropna().drop_duplicates().reset_index(drop=True)

	# generate mappings using pykeen function
	factory = TriplesFactory.from_labeled_triples(triples=triplets[['head', 'relation', 'tail']].values,
												  create_inverse_triples=False)

	# save mappings
	entity_to_id = factory.entity_to_id
	relation_to_id = factory.relation_to_id

	entity_to_id_file = triplets_file_utils.format(file_name="entity_to_id")
	save_json(entity_to_id, entity_to_id_file)

	relation_to_id_file = triplets_file_utils.format(file_name="relation_to_id")
	save_json(relation_to_id, relation_to_id_file)

	logger.info("creating pretrained entity embeddings")
	# Load BERT tokenizer and model
	model = SentenceTransformer('all-MiniLM-L6-v2')
	embeddings = model.encode(list(entity_to_id.keys()), show_progress_bar=True, device=device)
	numpy.save(pretrained_embedding_file, embeddings)


def hyperparameter_optimization(model_name: str,
								model_dir: str,
								noisy_triples_file: str,
								triplets_file_utils: str,
								pretrained_embedding_file: str = None):
	"""
	The hyperparameter_optimization function is used to optimize the hyperparameters of a model.

	:param model_name: str: Specify the model to be used for training
	:param model_dir: str: Save the model
	:param noisy_triples_file: str: Load the triples from a file
	:param triplets_file_utils: str: Load entity-to-id and relation-to-id mappings
	:param pretrained_embedding_file: file where the pretrained embeddings are saves
	:param ratio: float: Indicate the noise ratio of the dataset to be used
	"""
	ratio = 0  # only hypertrain on gold

	logger.info(f"## ===== HYPER-OPTIMIZATION TRAINING {model_name} on {ratio}% noise ratio ===== ##".upper())

	# get train, val, test
	train, test, val = get_train_val_test_from_dir(noisy_triples_file, noise=ratio, get_noisy_test=False)

	logger.info("RELATION COUNTS: ")
	logger.info(train['relation'].value_counts())

	train_factory, val_factory, test_factory = get_train_val_test_factory(train,
																		  val,
																		  test,
																		  triplets_file_utils,
																		  create_inverse_triples=False)

	if config.USE_PRETRAINED_EMBEDDINGS:
		assert pretrained_embedding_file is not None
		pretrained_embedding_tensor = torch.FloatTensor(np.load(pretrained_embedding_file))
		model_kwargs = dict(embedding_dim=pretrained_embedding_tensor.shape[-1],
							entity_initializer=PretrainedInitializer(tensor=pretrained_embedding_tensor), )
		model_kwargs_ranges = None
	else:
		model_kwargs = None
		model_kwargs_ranges = dict(embedding_dim=dict(type=int, low=5, high=150))
	hpo_results = hpo_pipeline(training=train_factory,
							   validation=val_factory,
							   testing=test_factory,
							   model=model_name,
							   model_kwargs=model_kwargs,
							   model_kwargs_ranges=model_kwargs_ranges,
							   # optimizer args
							   optimizer="Adam",
							   # training loop args
							   training_loop="slcwa",
							   negative_sampler="bernoulli",
							   negative_sampler_kwargs=dict(filterer='python-set', filtered=True),
							   # corruption_scheme=('head', 'relation', 'tail')),
							   # training args
							   training_kwargs={"use_tqdm_batch": False, },
							   training_kwargs_ranges=dict(num_epochs=dict(type=int, low=30, high=200, q=5),
														   batch_size=dict(type=int, low=64, high=256, q=64), ),
							   stopper=None,
							   # evaluation args
							   evaluator="RankBasedEvaluator",
							   evaluation_kwargs={
								   "use_tqdm"                 : True,
								   "additional_filter_triples": [train_factory.mapped_triples,
																 val_factory.mapped_triples, ], },
							   evaluator_kwargs={"filtered": True, },
							   metric="both.realistic.inverse_harmonic_mean_rank",
							   # MRR
							   filter_validation_when_testing=True,
							   # misc args
							   device=config.DEVICE,
							   # Optuna study args
							   sampler=TPESampler(consider_prior=True,
												  prior_weight=1.0,
												  consider_magic_clip=True,
												  consider_endpoints=False,
												  n_startup_trials=18,
												  n_ei_candidates=32, ),
							   pruner=PercentilePruner(percentile=70.0, n_startup_trials=5, ),
							   direction="maximize",
							   n_trials=config.NUM_TRIALS)

	logger.info(f"Best hyper-parameters: {hpo_results.study.best_params}")
	logger.info(f"## ===== HYPER-OPTIMIZATION TRAINING COMPLETE ===== ##".upper())

	logger.info(f"saving {model_name} to {model_dir}")
	hpo_results.objective.evaluation_kwargs['additional_filter_triples'] = None
	hpo_results.objective.model_kwargs = None
	hpo_results.save_to_directory(model_dir)
	gc.collect()


def main():
	generate_mappings(dataset, triplets_file_utils, pretrained_embedding_file)


if __name__ == '__main__':
	main()
