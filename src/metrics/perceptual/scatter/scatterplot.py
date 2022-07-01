from metrics.perceptual.plot import Plot
from environment.envconfig import Configuration

class ScatterPlot(Plot):

    def __init__(self, data):
        super(ScatterPlot, self).__init__(data)
        self.len_data = len(self.data)
        self.env_config = Configuration('res/config.json')

        self.default_width = self.env_config.get_property('default_width')
        self.default_height = self.env_config.get_property('default_height')

        self.kernel_sa = self.env_config.get_property("kernel_sa")
        self.delta = self.env_config.get_property("delta_sa")

    def collision_point_ratio(self):

        def filter_collisions(data):
            return data.drop_duplicates(subset=['x', 'y'], keep='first')

        data_points_no_duplicated = filter_collisions(self.data)
        return (self.len_data - len(data_points_no_duplicated))/self.len_data

    def point_pixel_ratio(self):
        return self.len_data / (self.default_height * self.default_width)

    def get_k(self):
        return self.len_data - self.data.drop_duplicates(subset=['x', 'y'], keep='first').shape[0]

    def precentage_screen_distorption(self, mode='bgsar'):
        psa = 0
        bgsar_ = 0
        sa_n = 0

        area = self.kernel_sa * self.kernel_sa

        for i in range(0, int(self.default_width), self.kernel_sa):

            for j in range(0, int(self.default_height), self.kernel_sa):

                sa = self.data.loc[(self.data['x'] > i) & (self.data['x'] < i + self.kernel_sa) & (self.data['y'] > j) & (self.data['y'] < j + self.kernel_sa)]
                sa_k = sa.duplicated().values.sum()

                sa_n += 1

                if (sa_k / area) >= self.delta:
                    if mode == 'bgsar':
                        bgsar_ += 1
                    elif mode == 'cppr':
                        psa += sa_k

        if mode == 'bgsar':
            return bgsar_/sa_n
        elif mode == 'cppr':
            return psa/self.len_data

        return -1