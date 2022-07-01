import json
from os.path import exists as file_exists
from utils.errors.errorhandler import Handler


class Configuration:

    error_handler = Handler('log_output.txt')

    def __init__(self, filename):
        if file_exists(filename):
            with open(filename, 'r') as f:
                self.params = json.load(f)
        else:
            Configuration.error_handler.error(f'File {filename} does not exist.')
            quit()


    def get_property(self, property_name):
        return self.params.get(property_name, None)

    def insert_property(self, property_name, value):
        self.params[property_name] = value