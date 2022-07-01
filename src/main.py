from utils.datareader.plot_reader import PlotReader
from utils.datawriter.plot_writer import PlotWriter
from metrics.perceptual.scatter.scatterplot import ScatterPlot
from metrics.perceptual.scatter.scagnostics.diagnostics import ScatterPlotDiagnostics

if __name__ == '__main__':

    plt_writer = PlotWriter('res/wine')
    plt_reader = PlotReader('src/wine.csv', plot_writer=plt_writer)
    plt_reader.process('points','price')
    data_coords = plt_reader.transform_data_in_pixel_coords()

    scatter_plt = ScatterPlot(data_coords)

    print(f'cpr = {scatter_plt.collision_point_ratio()}')
    print(f'ppr = {scatter_plt.point_pixel_ratio()}')
    print(f'get_k = {scatter_plt.get_k()}')
    # print(f'distorption_bgsar = {scatter_plt.precentage_screen_distorption()}')
    # print(f'distorption_cppr = {scatter_plt.precentage_screen_distorption(mode="cppr")}')

    print(f'bank_slopes_ms = {scatter_plt.bank_slopes()}')
    print(f'bank_slopes_avg = {scatter_plt.bank_slopes(method="avg")}')

    print(f'shape={data_coords.shape}')
    data_coords.drop_duplicates(subset=['x','y'], keep='first', inplace=True)
    print(f'shape={data_coords.shape}')
    scagnostic_plt = ScatterPlotDiagnostics(data_coords)
    scagnostics = scagnostic_plt.compute()


    print(scagnostics[0].bdata.point1)
    print(scagnostics[0].bdata.point2)
    print(scagnostics[0].bdata.count)

