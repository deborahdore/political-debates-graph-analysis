from config.config import dataset_file, noisy_dataset_file, original_dataset_file
from utils.dataset_utils import generate_noise, generate_triplets

if __name__ == '__main__':
	generate_triplets(original_dataset_file, dataset_file)

	generate_noise(dataset_file, noisy_dataset_file, noise_ratio=0.05)  # 5%
	generate_noise(dataset_file, noisy_dataset_file, noise_ratio=0.10)  # 10%
	generate_noise(dataset_file, noisy_dataset_file, noise_ratio=0.20)  # 20%
	generate_noise(dataset_file, noisy_dataset_file, noise_ratio=0.30)  # 30%
	generate_noise(dataset_file, noisy_dataset_file, noise_ratio=1)  # 100%

	print("END")
