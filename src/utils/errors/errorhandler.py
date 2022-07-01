import logging

class Handler:

    def __init__(self, log_file_path):
        logging.basicConfig(filename=log_file_path, filemode='a+', format='%(name)s - %(levelname)s - %(message)s')

    def debug(self, msg):
        logging.debug(msg)

    def error(self, msg):
        logging.error(msg)
