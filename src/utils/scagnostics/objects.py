from curses.ascii import NUL
from pyparsing import null_debug_action
from environment.envconfig import Configuration
import math

class BinnedData:

    def __init__(self, point1, point2, counts):
        self.env_config = Configuration('res/config.json')
    
        self.resolution = self.env_config.get_property('resolution')

        self.point1 = self.__integerise_data(point1)
        self.point2 = self.__integerise_data(point2)
        self.counts = counts

    def __integerise_data(self, data):
        return [self.resolution * data[i] for i in range(len(data))]

class Binner:

    def __init__(self, max_bins):
        self.max_bins = max_bins

    def bin_hex(self, point1, point2, num_bins):
        con1 = .25
        con2 = 1 / 3
        c1 = num_bins - 1
        c2 = c1 / math.sqrt(3)

        jinc = num_bins
        iinc = 2 * num_bins
        num_bin = (num_bins + 20)**2

        count = [0] * num_bin
        xbin = [0] * num_bin
        ybin = [0] * num_bin

        for i in range(len(point1)):
            if not (math.isnan(point1[i]) or math.isnan(point2[i])):
                sx, sy = c1 * point1[i], c2 * point2[i]
                i1, j1 = int(sy + .5), int(sx + .5)
                dy, dx = sy - i1, sx - j1
                dist1 = dx * dx + 3 * dy * dy
                m = -1
                if dist1 < con1:
                    m = i1 * iinc + int(sx) + jinc
                elif dist1 > con2:
                    m = int(sy) * iinc + int(sx) + jinc
                else:
                    i2 = int(sy)
                    j2 = int(sx)
                    dy, dx = sy - i2 - .5, sx - j2 - .5
                    dist2 = dx * dx + 3 * dy * dy
                    if dist1 <= dist2:
                        m = i1 * iinc + j1
                    else:
                        m = i2 * iinc + j2 + jinc

                count[m] += 1
                xbin[m] += (point1[i] - xbin[m]) / count[m]
                ybin[m] += (point2[i] - ybin[m]) / count[m]

        num_bin = self.__delete_empty_bins(count, xbin, ybin)
        if num_bin > self.max_bins:
            num_bins = 2 * num_bins / 3
            return self.bin_hex(point1, point2, num_bins)
      
        tcount = count[:num_bin]
        xtbin = xbin[:num_bin]
        ytbin = ybin[:num_bin]

        return BinnedData(xtbin, ytbin, tcount)


    def __delete_empty_bins(self, count, xbin, ybin):
        k = 0
        for i in range(len(count)):
            if count[i] > 0:
                count[k] = count[i]
                xbin[k] = xbin[i]
                ybin[k] = ybin[i]
                k += 1

        return k


class Node:

    def __init__(self, x, y, count, point1D):
        self.x = x
        self.y = y
        self.count = count
        self.point1D  = point1D
        self.anEdge = None
        self.neighbors = list()
        self.onHull = False
        self.isVisited = False

    def dist_to_node(self, px, py):
        dx = px - self.x
        dy = py - self.y
        return math.sqrt(( dx**2 + dy**2))

    def set_neighbor(self, neighbor):
        self.neighbors.append(neighbor)
    
    def get_neighbor_iterator(self):
        return self.neighbors
    
    def shortest_edge(self, mst):
        emin = None
        if(self.neighbors != None):
            DOUBLE_MAX = 1.7976931348623158E+308 
            for edge in self.neighbors:
                e =  Edge(edge)
                if( mst or not e.otherNode(self).onMST):
                    wt = e.weight
                    if( wt < DOUBLE_MAX):
                        wmin = wt
                        emin = e
        return emin


    def get_MST_Children(self,cutoff, max_lenght):
        count = 0
        if(self.visited):
            return count
        self.isVisited = True

        for edge in self.neighbors:
            e =  Edge(edge)
            if(e.onMST and e.weight < cutoff and not e.otherNode(self).isVisited):
                count += e.otherNode(self).getMSTChildren(cutoff, max_lenght)
                el = e.weight
                if(el > max_lenght[0]):
                    max_lenght[0] = el

        count += self.count
        return count









class Scagnostic:

    def __init__(self, point1, point2, num_bins, max_bins):
        self.point1 = point1
        self.point2 = point2
        self.num_bins = num_bins
        self.max_bins = max_bins


        self.nodes = list()
        self.edges = list()

        self.triangles = list()
        self.mst_edges = list()

        self.b = Binner(self.max_bins)
        self.bdata = self.b.bin_hex(self.point1, self.point2, self.num_bins)


