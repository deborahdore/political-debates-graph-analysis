import json

from loguru import logger


def save_json(json_obj: json, json_file_name: str):
	"""
	The save_json function takes a json object and saves it to the specified file name.


	:param json_obj: json: Specify the type of data that is being passed into the function
	:param json_file_name: str: Specify the name of the json file to be written
	:return: Nothing
	"""
	logger.info("[write_json] writing json file")
	with open(json_file_name, 'w') as file:
		file.write(json.dumps(json_obj, indent=4))
	file.close()
