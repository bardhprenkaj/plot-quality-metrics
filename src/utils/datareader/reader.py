from abc import ABC, abstractmethod
import pandas as pd
from os.path import exists as file_exists
from os.path import basename as extract_name
import logging

from utils.errors.errorhandler import Handler

class Reader(ABC):

    error_handler = Handler('log_output.txt')

    def __init__(self, file_path, sep=','):
        self.file_path = file_path

        if file_exists(self.file_path):
            self.data_frame = pd.read_csv(self.file_path, sep=sep, header=0)
            self.filename = self.__extract_name(self.file_path)
        else:
            Reader.error_handler.error(f'File {self.filename} does not exist.')
            quit()


    @abstractmethod
    def visualise_data(self, **kwargs):
        pass

    @abstractmethod
    def process(self, feature_x, feature_y):
        pass


    def __extract_name(self, file_path):
        return extract_name(file_path)



    