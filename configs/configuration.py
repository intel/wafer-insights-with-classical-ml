import yaml
import os

def get_config():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + "/config.yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        configs = yaml.load(file, Loader=yaml.FullLoader)
        return configs
