from os.path import join
from utils.datawriter.writer import Writer

class PlotWriter(Writer):

    def __init__(self, file_path):
        super(PlotWriter, self).__init__(file_path)

    def write(self, figure, file_name):
        figure.savefig(join(self.file_path, file_name))
