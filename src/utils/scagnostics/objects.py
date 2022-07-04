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

class Triangle:

    def __init__(self, e1, e2, e3):
        self.edge = None
        self.centre_x = None
        self.centre_y = None
        self.radius = None
        self.on_complex = True

        self.update(e1, e2, e3)


    def update(self, e1, e2, e3):
        edge = e1
        e1.set_next_edge(e2)
        e2.set_next_edge(e3)
        e3.set_next_edge(e1)

        e1.set_triangle(self)
        e2.set_triangle(self)
        e3.set_triangle(self)

        self.find_circle(self)

    def find_circle(self):
        pass

    def in_circle(self, node):
        return node.dist_to_node(self.centre_x, self.centre_y) < self.radius

    def remove_edges(self, edges):
        edges.remove(self.edge)
        edges.remove(self.edge.next_edge)
        edges.remove(self.edge.next_edge.next_edge)

class Edge:

    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

        self.next_edge = None
        self.inverse_edge = None
        self.next_convect_hull_link = None

        self.triangle = None

        self.a, self.b, self.c = (0,0,0)

        self.on_hull = False
        self.on_mst = False
        self.on_shape = False
        self.on_outlier = False

        self.update(node1, node2)

    def set_next_edge(self, next_edge):
        self.next_edge = next_edge

    def set_triangle(self, triangle):
        self.triangle = triangle

    def update(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

        self.a = node2.y - node1.y
        self.b = node1.x - node2.x
        self.c = node2.x * node1.y - node1.x * node2.y

        self.weight = math.sqrt(self.a * self.a + self.b * self.b)

        self.as_index()

    def as_index(self):
        self.node1.edge = self

    def make_symm(self):
        edge = Edge(self.node2, self.node1)
        self.link_symm(edge)
        return edge

    def link_symm(self, edge):
        self.inverse_edge = edge
        if edge:
            edge.inverse_edge = self 

    def on_side(self, node):
        s = self.a * node.x + self.b * node.y + c
        if s > 0:
            return 1
        if s < 0:
            return -1
        return 0

    def most_left(self):
        e = self
        ee = self
        ee = e.next_edge.next_edge.inverse_edge
        while ee and not self.is_equal(ee):
            e = ee
        return e.next_edge.next_edge

    def most_right(self):
        e = self
        ee = self
        ee = e.inverse_edge.next_edge
        while e.inverse_edge and ee:
            e = ee
        return e

    def delete_simplex(self):
        on_shape = False
        self.triangle.on_complex = False
        if self.inverse_edge:
            self.inverse_edge.on_shape = False
            self.inverse_edge.triangle.on_complex = False


    def is_equal(self, other):
        return other.node1.x == self.node1.x and other.node2.x == self.node2.x and other.node1.y == self.node1.y and other.node2.y == self.node2.y

    def is_equivalent(self, other):
        return self.is_equal(other) and (other.node1.x == self.node2.x and other.node1.y == self.node2.y and other.node2.x == self.node1.x and other.node2.y == self.node1.y)

    def other_node(self, n):
        return self.node2 if n.equals(self.node1) else self.node1

    def is_new_edge(self, n):
        for edge in n.neighbors:
            if self.is_equivalent(edge):
                return False
        return True

    def get_runts(self, max_length):
        max_length1 = 0
        max_length2 = 0

        count1 = self.node1.get_MST_Children(self.weight, max_length1)
        count2 = self_node2.getMSTChildren(self.weight, max_length2)

        if count1 < count2:
            max_length[0] = max_length1
            return count1
        elif count1 == count2:
            max_length2[0] = max_length1 if max_length1 < max_length2 else max_length2
            return count1
        else:
            max_length[0] = max_length2
            return count2

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

        self.env_config = Configuration('res/config.json')

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
        wmin = self.env_config.get_property("double_max")
        if self.neighbors:
            for edge in self.neighbors:
                if mst or edge.is_equal(self.neighbors[-1]):
                    wt = edge.weight
                    if wt < wmin:
                        wmin = wt
                        emin = edge
        return emin


    def 






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


