from abc import ABC, abstractmethod
from os import makedirs

from os.path import exists as file_exists
from os.path import basename as extract_name

from utils.errors.errorhandler import Handler

class Writer(ABC):

    error_handler = Handler('log_output.txt')

    def __init__(self, file_path, sep=','):
        self.file_path = file_path

        if file_exists(self.file_path):
            self.filename = self.__extract_name(self.file_path)
        else:
            makedirs(self.file_path, exist_ok=True)

    def __extract_name(self, file_path):
        return extract_name(file_path)