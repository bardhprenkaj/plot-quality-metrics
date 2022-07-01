from abc import ABC
import numpy as np

class Plot(ABC):

    def __init__(self, data):
        self.data = data

    def __calc_slope(self, cull=False):
        x = self.data['x'].values
        y = self.data['y'].values

        dx = np.absolute(np.diff(x))
        dy = np.diff(y)

        s = dy / dx

        mask = np.all(np.absolute(s) > 0 & np.isfinite(s)) if cull else np.isfinite(s)

        return [s[mask],
                dx[mask],
                dy[mask],
                x.max() - x.min(),
                y.max() - y.min()
        ]

    def bank_slopes(self, cull=False, method='ms'):

        def medianSlope(vals):
            return np.median(np.absolute(vals[0])) * vals[3] / vals[4]

        def averageSlope(vals):
            return np.mean(np.absolute(vals[0])) * vals[3] / vals[4]

        slope = self.__calc_slope()
        xyrat = medianSlope(slope) if method == 'ms' else averageSlope(slope)

        return 1 / xyrat