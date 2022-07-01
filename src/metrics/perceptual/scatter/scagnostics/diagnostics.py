from metrics.perceptual.scatter.scatterplot import ScatterPlot
from environment.envconfig import Configuration
from utils.scagnostics.objects import Scagnostic
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class ScatterPlotDiagnostics(ScatterPlot):

    def __init__(self, data):
        super(ScatterPlotDiagnostics, self).__init__(data)

        self.__normalise_data()

        self.env_config = Configuration('res/config.json')

        self.num_bins = self.env_config.get_property('num_bins')
        self.max_bins = self.env_config.get_property('max_bins')

    def compute(self):
        scagnostics = []

        for i in range(self.len_data):
            for j in range(i):
                scagnostic = Scagnostic(self.data.values[j], self.data.values[i], self.num_bins, self.max_bins)
                scagnostics.append(scagnostic)#.compute()

        return scagnostics


    def __normalise_data(self):
        self.data['x'] = MinMaxScaler().fit_transform(np.array(self.data['x']).reshape(-1,1))
        self.data['y'] = MinMaxScaler().fit_transform(np.array(self.data['y']).reshape(-1,1))