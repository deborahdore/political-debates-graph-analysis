import gc
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger
from pykeen.hpo import hpo_pipeline
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from utils.utils import read_json


def training(model_dir: str,
			 model_name: str,
			 plot_dir: str,
			 train_factory: TriplesFactory,
			 test_factory: TriplesFactory,
			 val_factory: TriplesFactory,
			 ratio: float):
	pipeline_config = model_dir + f"/{ratio}/best_pipeline/pipeline_config.json"
	pipeline_config = read_json(pipeline_config)['pipeline']

	logger.info(f"starting pipeline --> {model_name} with ratio {ratio}")
	logger.info(f"{pipeline_config}")

	result = pipeline(training=train_factory,
					  testing=test_factory,
					  validation=val_factory,
					  model=model_name,
					  evaluator=pipeline_config['evaluator'],
					  filter_validation_when_testing=pipeline_config['filter_validation_when_testing'],
					  loss=pipeline_config['loss'],
					  loss_kwargs=pipeline_config['loss_kwargs'],
					  model_kwargs=pipeline_config['model_kwargs'],
					  negative_sampler=pipeline_config['negative_sampler'],
					  negative_sampler_kwargs=pipeline_config['negative_sampler_kwargs'],
					  optimizer=pipeline_config['optimizer'],
					  optimizer_kwargs=pipeline_config['optimizer_kwargs'],
					  training_kwargs=pipeline_config['training_kwargs'],
					  training_loop=pipeline_config['training_loop'],
					  use_tqdm=False,
					  random_seed=42, )

	model_file = os.path.join(model_dir, f"{model_name}_{ratio}.pt")
	logger.info(f"{model_name} training complete")
	logger.info(f"saving {model_name} to {model_file}")
	result.save_model(path=model_file)

	result.plot_losses()

	plot_file = os.path.join(plot_dir, f"{model_name}_{ratio}_loss.svg")
	plt.savefig(plot_file)
	gc.collect()

	return result


def hyperparameter_optimization(model_name: str,
								model_dir: str,
								train_factory: TriplesFactory,
								test_factory: TriplesFactory,
								val_factory: TriplesFactory,
								ratio: float):
	logger.info(f"starting optimizer pipeline - {model_name} with ratio {ratio}")

	# otherwise TransH won't work
	regularizer = "OrthogonalityRegularizer" if model_name == 'TransH' else None

	hpo_results = hpo_pipeline(training=train_factory,
							   testing=test_factory,
							   validation=val_factory,
							   model=model_name,
							   n_trials=15,
							   regularizer=regularizer,
							   optimizer="Adam",
							   optimizer_kwargs_ranges=dict(lr=dict(type=float, low=0.0001, high=0.01, scale="log"), ),
							   training_loop="slcwa",
							   training_kwargs_ranges=dict(num_epochs=dict(type=int, low=30, high=200, q=5),
														   batch_size=dict(type=int, low=64, high=256, q=64), ),
							   negative_sampler="basic",
							   metric="both.realistic.inverse_harmonic_mean_rank",
							   stopper=None,
							   evaluator="RankBasedEvaluator",
							   evaluation_kwargs={
								   "use_tqdm"                 : True,
								   "additional_filter_triples": [train_factory.mapped_triples,
																 val_factory.mapped_triples], },
							   filter_validation_when_testing=True, )

	logger.info(f"model {model_name} training complete")

	model_dir_ratio = model_dir + f"/{ratio}"
	logger.info(f"saving {model_name} to {model_dir_ratio}")

	Path(model_dir_ratio).mkdir(exist_ok=True, parents=True)
	hpo_results.save_to_directory(model_dir_ratio)
	gc.collect()
