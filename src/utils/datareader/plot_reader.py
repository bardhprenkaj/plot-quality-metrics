from utils.datareader.reader import Reader
from environment.envconfig import Configuration
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd

class PlotReader(Reader):

    def __init__(self, filename, plot_writer=None):
        super(PlotReader, self).__init__(filename)
        self.env_config = Configuration('res/config.json')
        self.plot_writer = plot_writer

    def process(self, feature_x, feature_y):
        self.data_frame = self.data_frame[[feature_x, feature_y]]
        self.data_frame.columns = ['x','y']
        self.data_frame.dropna(inplace=True)
        self.data_frame = self.data_frame.astype('float64')

    def transform_data_in_pixel_coords(self):
        x, y, h, w = self.visualise_data()

        self.env_config.insert_property('default_height', h)
        self.env_config.insert_property('default_width', w)

        return pd.DataFrame(list(zip(x, y)), columns = ['x', 'y'])

    def visualise_data(self, **kwargs):
        style = {}

        marker_size = kwargs.get('marker_size', None)
        style['markersize'] = marker_size if marker_size else self.env_config.get_property('marker_size')

        plt.rcParams['figure.figsize'] = self.env_config.get_property('figsize')
        plt.rcParams['figure.autolayout'] = self.env_config.get_property('autolayout')

        fig, ax = plt.subplots()

        id1 = kwargs.get('id1', None)
        id1 = id1 if id1 else self.env_config.get_property('id1')

        id2 = kwargs.get('id2', None)
        id2 = id2 if id2 else self.env_config.get_property('id2')

        points, = ax.plot(np.asarray(self.data_frame[id1]), np.asarray(self.data_frame[id2]), '.', **style)

        x, y = points.get_data()

        ax.get_xbound()
        pixels = ax.transData.transform(np.vstack([x,y]).T)
        x, y = pixels.T

        _, height = fig.canvas.get_width_height()
        y = height - y

        bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        bbox_h, bbox_w = bbox.width * fig.dpi, bbox.height * fig.dpi

        if self.plot_writer:
            self.plot_writer.write(fig, f'plot_{self.filename}-{datetime.now()}.png')

        fig.show()

        return x, y, bbox_h, bbox_w