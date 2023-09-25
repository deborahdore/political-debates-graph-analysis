from config.config import original_dataset_file, dataset_file
from utils.dataset_utils import generate_triplets

if __name__ == '__main__':
	generate_triplets(original_dataset_file, dataset_file)
