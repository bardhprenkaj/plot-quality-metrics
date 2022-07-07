from asyncio import coroutines
from asyncio.format_helpers import _format_callback_source
from curses.ascii import NUL
from stat import FILE_ATTRIBUTE_SPARSE_FILE
from pyparsing import null_debug_action
import random
from environment.envconfig import Configuration
import math
import numpy as np

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

        self.find_circle()

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
        s = self.a * node.x + self.b * node.y + self.c
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
        count2 = self.node2.get_MST_Children(self.weight, max_length2)

        if count1 < count2:
            max_length[0] = max_length1
            return count1
        elif count1 == count2:
            max_length2 = max_length1 if max_length1 < max_length2 else max_length2
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
        self.mst_degree = 0
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


    def get_MST_Children(self, cutoff, max_length):
        count = 0
        if(self.isVisited):
            return count
        self.isVisited = True

        for e in self.neighbors:
            if(e.on_mst and e.weight < cutoff and not e.other_node(self).isVisited):
                count += e.other_node(self).get_MST_Children(cutoff, max_length)
                el = e.weight
                if(el > max_length[0]):
                    max_length[0] = el

        count += self.count
        return count


class Cluster:

    def __init__(self, num_clusters, num_iterations):
        self.members = np.array(list())
        self.n_var = None
        self.n_row = None
        self.num_clusters = 0
        self.num_iterations = 3

        if num_iterations != 0:
            self.num_iterations = num_iterations

        if num_clusters != 0:
            self.num_clusters = num_clusters


    def compute(self, data):
        def init_array(x, y):
            return np.array([np.array([0] * x)] * y)

        self.n_row, self.n_var = data.shape


        use_stopping_rule = False
        
        ssr = list()
        if self.num_clusters == 0:
            use_stopping_rule = True
            self.num_clusters = 25
            ssr = init_array(self.n_var, self.num_clusters)


        center = init_array(self.n_var, self.num_clusters)
        count = init_array(self.n_var, self.num_clusters)
        mean = init_array(self.n_var, self.num_clusters)
        min_ = init_array(self.n_var, self.num_clusters)
        max_ = init_array(self.n_var, self.num_clusters)
        ssq = init_array(self.n_var, self.num_clusters)
        closest_points = [0] * self.num_clusters
        closest_distances = [0] * self.num_clusters

        self.members = np.array([0] * self.n_row)
        mem = np.array([0] * self.n_row)


        for k in range(self.num_clusters):

            for iter in range(self.num_iterations):

                reassigned = False

                for l in range(k):
                    for j in range(self.n_var):
                        if iter == 0 or center[l][j] != mean[l][j]:
                            self.reassign(k, data, center, count, mean, min_, max_, ssq, closest_points, closest_distances)
                            reassigned = True
                            break
                if reassigned:
                    break
            
            if not reassigned or k == 0:
                break


            if use_stopping_rule:
                ssq1, ssq2 = (0,0)
                for j in range(self.n_var):
                    for l in range(k):
                        ssq1 += ssr[l][j]
                        ssq2 += ssq[l][j]
                        ssr[l][j] = ssq[l][j]

                pre = (ssq1 - ssq2) / ssq1
                if pre > 0 and pre < .1:
                    self.num_clusters = k
                    self.reassign(k, data, center, count, mean, min_, max_, ssq, closest_points, closest_distances)
                    mem[0:self.n_row] = self.members[0:self.n_row]
                    break
                else:
                    self.members[0:self.n_row] = mem[0:self.n_row]


            if k < (self.num_clusters - 1):
                kn, dmax, jm, km, cutpoint = (k + 1, 0, 0, 0, 0)
                for l in range(k):
                    for j in range(self.n_var):
                        dm = max_[l][j] - min_[l][j]
                        if dm > dmax:
                            cutpoint = mean[l][j]
                            dmax = dm
                            jm = j
                            km = l
                for i in range(self.n_row):
                    if self.members[i] == km and data[i][jm] > cutpoint:
                        for j in range(self.n_var):
                            count[km][j] -= 1
                            count[kn][j] += 1
                            mean[km][j] += (data[i][j] - mean[km][j]) / count[km][j]
                            mean[kn][j] += (data[i][j] - mean[kn][j]) / count[kn][j]

        nc, cutoff = 0, .1
        for k in range(self.num_clusters):
            if (count[k][0] / self.n_row) > cutoff:
                nc += 1

        exemplars = [0] * nc
        nc = 0
        for k in range(self.num_clusters):
            if (count[k][0] / self.n_row) > cutoff:
                exemplars[nc] = closest_points[k]
                nc += 1

        return exemplars


    def reassign(self, num_clusters, data, center, count, mean, min_, max_, ssq, closest_points, closest_distances):

        for k in range(num_clusters):
            closest_points[k] = -1
            closest_distances[k] = float('inf')
            for j in range(self.n_var):
                center[k][j] = mean[k][j]
                mean[k][j] = 0
                count[k][j] = 0
                ssq[k][j] = 0
                min_[k][j] = float('inf')
                max_[k][j] = float('-inf')

        for i in range(self.n_row):
            dmin = float('inf')
            kmin = -1
            for k in range(num_clusters):
                dd = self.distance(data[i], center[k])
                if dd < dmin:
                    dmin = dd
                    kmin = k
                    if dmin < closest_distances[k]:
                        closest_distances[k] = dmin
                        closest_points[k] = i

            if kmin < 0:
                self.members[i] = -1
            else:
                self.members[i] = kmin

            
            for j in range(self.n_var):
                if data[i][j]:
                    count[kmin][j] += 1
                    xn = count[kmin][j]
                    xa = data[i][j]
                    mean[kmin][j] += (xa - mean[kmin][j]) / xn

                    if xn > 1:
                        ssq[kmin][j] += xn * (xa - mean[kmin][j]) * (xa - mean[kmin][j]) / (xn - 1)
                    if min_[kmin][j] > xa:
                        min_[kmin][j] = xa
                    if max_[kmin][j] < xa:
                        max_[kmin][j] = xa


    def distance(self, a, b):
        dist = 0
        for i in range(len(a)):
            dist += (a[i] - b[i]) * (a[i] - b[i])
        return dist

class Sorts:


    def __init__(self) -> None:
        pass



    def double_array_sort(x, from_index, to_index):
        if(from_index == to_index):
            from_index = 0
            to_index = len(x)
        x[from_index:to_index] = sorted(x[from_index:to_index])
    
    @classmethod
    def indexed_double_array_sort(x, from_index, to_index):
        def sort(x):
            for i in range(len(x)):
                for j in range(len(x)-i-1):
                    if x[int(x[j])] >= x[int(x[j+1])]:
                        x[j], x[j+1] = x[j+1], x[j]
            return x
            
        if(from_index == to_index):
            from_index = 0
            to_index = len(x)
        sort_order = [None] * (to_index - from_index)
        for i in range(len(sort_order)):
            sort_order[i] = i
        x[from_index:to_index] = sort(x[from_index:to_index])

        return [int(i) for i in x]
    
    def rank(self, a):
        ranks = [0] * len(a)
        index = self.indexed_double_array_sort(a, 0, len(a))
        lind = index[0]
        am = a[lind]
        k1, k2, ak, kms = (0, 0, 1.0, 1)


        for k in range(len(a)):
            kind = index[k]
            insert = True
            if not math.isnan(am):
                freq = 1.0
                kms += int(freq)
                if freq > 1.0:
                    ak += 0.5 * (freq - 1.0)
                    k1, k2 = (k, k)
                elif a[kind] == am:
                    k2 = k
                    ak += 0.5
                    if k < (len(a) - 1):
                        insert = False
                
                if(insert):
                    for l in range(k1, k2):
                        lind = index[l]
                        ranks[lind] = ak
                    if(k2 != len(a) -1 and k == len(a)-1):
                        ranks[kind] = kms
            if(insert):
                k1, k2, ak, am = (k, k, kms, a[kind])
               
        return ranks

                
        
class Scagnostic:

    def __init__(self, point1, point2, num_bins, max_bins):
        self.env_config = Configuration('res/config.json')
    
        self.point1 = point1
        self.point2 = point2
        self.num_bins = num_bins
        self.max_bins = max_bins


        self.nodes = list()
        self.edges = list()

        self.triangles = list()
        self.mst_edges = list()
        self.hull_start = None
        self.act_e = None
        
        self.total_peeled_count = 0
        self.total_count = 0
        
        self.alpha_area, self.alpha_perimeter, self.hull_area, self.hull_perimeter = (1,1,1,1)
        
        self.total_original_mst_lengths = 0
        
        self.total_mst_outlier_lengths = 0
        
        self.sorted_original_mst_lengths = list()
        
        self.num_stagnostics = 9

        self.scagnostic_labels = self.env_config.get_property('scagnostic_labels')
        self.outlying = self.env_config.get_property('outlying')
        self.skewed = self.env_config.get_property('skewed')
        self.clumpy = self.env_config.get_property('clumpy')
        self.sparse = self.env_config.get_property('sparse')
        self.striated = self.env_config.get_property('striated')
        self.convex = self.env_config.get_property('convex')
        self.skinny = self.env_config.get_property('skinny')
        self.stringy = self.env_config.get_property('stringy')
        self.monotonic = self.env_config.get_property('monotonic')

        self.fuzz = self.env_config.get_property('fuzz')
        
        self.is_outlier = list()
        self.px = list()
        self.py = list()
        self.counts = list()
        
        self.b = Binner(self.max_bins)
        self.bdata = self.b.bin_hex(self.point1, self.point2, self.num_bins)


    def compute(self):
        self.px = self.bdata.point1
        self.py = self.bdata.point2
        
        if len(self.px) < 3:
            return None
        
        xx = self.px[0]
        yy = self.py[0]
        
        is_x_constant, is_y_constant = True, True
        
        for i in range(1, len(self.px)):
            if self.px[i] != xx:
                is_x_constant = False
            if self.py[i] != yy:
                is_y_constant = False
                
        if is_x_constant or is_y_constant:
            return None
        
        
        self.find_outliers()
        
        self.computer_alpha_graph()
        self.compute_total_count()
        self.compute_alpha_area()
        self.compute_alpha_perimeter()
        self.compute_hull_area()
        self.compute_hull_perimeter()
        
        return self.compute_measures()
    
    
    def get_num_scagnostics(self):
        return len(self.scagnostic_labels)
    
    def get_scagnostics_labels(self):
        return self.scagnostic_labels
    
    
    def compute_scagnostics_exemplars(self, pts):
        if len(pts) < 2:
            return None
        
        c = Cluster(0, 0)
        exmp = c.compute(pts)
        exemplars = [False] * len(pts)
        
        for i in range(len(exmp)):
            exemplars[exmp[i]] = True
            
        return exemplars

    def compute_scagnostics_outliers(self, pts):
        npts = len(pts)
        nvar = len(pts[0])
        
        if npts < 2:
            return None
        
        edges = [[None] * (npts - 1)] * 2
        lst = [None] * npts
        degrees = [0] * npts
        cost = [0] * npts
        lengths = [0] * (npts - 1)
        
        lst[0] = 0
        cost[0] = float('inf')
        cheapest = 0
        
        for i in range(1, npts):
            for j in range(nvar):
                d = pts[i][j] - pts[0][j]
                cost[i] += d * d
            if cost[i] < cost[cheapest]:
                cheapest = i
                
                
        for j in range(1, npts):
            end = lst[cheapest]
            jp = j - 1
            edges[jp][0] = cheapest
            edges[jp][1] = end
            lengths[jp] = cost[cheapest]
            degrees[cheapest] += 1
            degrees[end] += 1
            cost[cheapest] = float('inf')
            end = cheapest
            
            for i in range(1, npts):
                if cost[i] != float('inf'):
                    dist = 0
                    for k in range(nvar):
                        d = pts[i][k] - pts[end][k]
                        dist += d*d
                    if dist < cost[i]:
                        lst[i] = end
                        cost[i] = dist
                        
                    if cost[i] < cost[cheapest]:
                        cheapest = i
                        
        cutoff = self.find_cutoff(lengths)
        outliers = [True for i in range(npts)]
        
        for i in range(npts-1):
            if lengths[i] < cutoff:
                for k in range(2):
                    node = edges[i][k]
                    outliers[node] = False
                    
                    
        return outliers
    
    
    def clear(self):
        self.nodes.clear()
        self.edges.clear()
        self.triangles.clear()
        self.mst_edges.clear()
        
        
    def find_outliers(self):
        self.counts = self.bdata.counts
        self.is_outlier = [False] * len(self.px)
        self.compute_dt()
        self.compute_mst()
        
        self.sorted_original_mst_lengths = self.get_sorted_mst_edge_lens()
        cutoff = self.compute_cutoff(self.sorted_original_mst_lengths)
        self.computer_total_original_mst_lens()
        
        found_new_outliers = self.compute_mst_outliers(cutoff)
        sorted_peeled_mst_lengths = None
        while found_new_outliers:
            self.clear()
            self.compute_dt()
            self.compute_mst()
            self.sorted_original_mst_lengths = self.get_sorted_mst_edge_lens()
            cutoff = self.compute_cutoff(sorted_peeled_mst_lengths)
            found_new_outliers = self.compute_mst_outliers(cutoff)
            
            
    def compute_total_count(self):
        return sum(self.counts)
    
    def compute_measures(self):
        result = [0] * self.num_stagnostics
        
        result[self.outlying] = self.compute_outlier_measure()
        result[self.clumpy] = self.compute_cluster_measure()
        result[self.skewed] = self.compute_mst_edge_len_skewness_measure()
        result[self.convex] = self.compute_convexity_measure()
        result[self.skinny] = self.compute_skinny_measure()
        result[self.stringy] = self.compute_stringy_measure()
        result[self.striated] = self.compute_striation_measure()
        result[self.sparse] = self.compute_sparseness_measure()
        result[self.monotonic] = self.compute_monotonicity_measure()

        return result
    
    
    def compute_dt(self):
        self.total_peeled_count = 0
        choices = list(range(0,13579))
        for i in range(len(self.px)):
            x = self.px[i] + int(8 * (random.choice(choices) - .5))
            y = self.py[i] + int(8 * (random.choice(choices) - .5))
            if not self.is_outlier[i]:
                self.insert(x, y, self.counts[i], i)
                self.total_peeled_count += self.counts[i]
                
        self.set_neighbours()
        self.mark_hull()
        
        
    def compute_mst(self):
        if len(self.nodes) > 1:
            mst_nodes = list()
            mst_node = self.nodes[0]
            self.update_mst_nodes(mst_node, mst_nodes)
            
            for i in range(len(self.nodes)):
                add_edge = None
                wmin = float('inf')
                nmin = None
                
                for mst_node in mst_nodes:
                    candidate_edge = mst_node.shortest_edge(False)
                    if candidate_edge:
                        wt = candidate_edge.weight
                        
                        if wt < wmin:
                            wmin, nmin, add_edge = wt, mst_node, candidate_edge
                            
                if add_edge:
                    add_node = add_edge.other_node(nmin)
                    self.update_mst_nodes(add_node, mst_nodes)
                    self.update_mste_edges(add_edge, self.mst_edges)
                    
    def find_cutoff(self, distances):
        index = Sorts.indexed_double_array_sort(distances, 0, 0)
        n50 = len(distances) / 2
        n25 = n50 / 2
        n75 = n50 + n25
        
        return distances[index[n75]] + 1.5 * (distances[index[n75]] - distances[index[n25]])
    
    
    def compute_mst_outliers(self, omega):
        found = False
        
        for node in self.nodes:
            delete = True
            for edge in node.neighbors:
                if edge.on_mst and edge.weight < omega:
                    delete = False
                    
            if delete:
                sum_length = 0
                for edge in node.neighbors:
                    if edge.on_mst and not edge.on_outlier:
                        sum_length += edge.weight
                        edge.on_outlier = True
                        
                self.total_mst_outlier_lengths += sum_length
                self.is_outlier[node.point1D] = True
                found = True
                
        return found
    
    def compute_cutoff(self, lengths):
        if len(lengths) == 0:
            return 0
        
        n50 = len(lengths) / 2
        n25 = n50 / 2
        n75 = n50 + n25
        
        return lengths[n75] + 1.5 * (lengths[n75] - lengths[n25])
    
    def compute_alpha_value(self):
        length = len(self.sorted_original_mst_lengths)
        if length == 0:
            return 100
        n90 = (9 * length) / 10
        alpha = self.sorted_original_mst_lengths[n90]
        return min(alpha, 100)
    
    def compute_mst_edge_len_skewness_measure(self):
        n = len(self.sorted_original_mst_lengths)
        if n == 0:
            return 0
        
        n50 = n / 2
        n10 = n / 10
        n90 = (9 * n) / 10
        
        skewness = (self.sorted_original_mst_lengths[n90] - self.sorted_original_mst_lengths[n50]) / (self.sorted_original_mst_lengths[n90] - self.sorted_original_mst_lengths[n10])
        t = self.total_count / 500
        correction = .7 + .3 / (1 + t * t)
        
        return 1 - correction * (1 - skewness)


    def update_mste_edges(self, add_edge, mst_edges):
       
        mst_edges.append(add_edge)
        add_edge.on_mst = True
        add_edge.node1.mst_degree +=1
        add_edge.node1.mst_degree +=1
    
    def update_mst_nodes(self, add_node, mst_nodes):
        mst_nodes.append(add_node)
        add_node.on_mst = False

    def get_shortest_mst_edge_lenghts(self):
        length = self.compute_edge_lengths(self.mst_edges, len(self.mst_edges))
        Sorts.double_array_sort(length, 0, 0)
        return length
    
    def compute_outliner_measure(self):
        return self.total_mst_outlier_lengths / self.total_original_mst_lengths
    
    def compute_edge_lengths(graph, n):
        length = [0] * n
        for index, e in graph:
            length[index] = e.weight
    
    def compute_points_in_circle(self, n, xc, yc, radius):
        r = self.env_config.get_property('FUZZ')
        i = n.neighbors
        for e in i:
            no = e.other_node(n)
            dist = no.dist_to_node(xc, yc)
            if(dist < r):
                return True
        return False
    
    def compute_alpha_graph(self):
        deleted = True
        alpha = self.compute_alpha_value()
        while(deleted):
            deleted = False
            for i in self.edges:
                e = i
                if(e.triangle.on_complex):
                    if(alpha < e.weight / 2):
                        e.triangle.on_complex = False
                        deleted = True
                    else:
                        if(e.inverse_edge != None and e.inverse_edge.triangle.on_complex):
                            continue
                        if(not self.edge_is_exposed(alpha, e)):
                            e.triangle.on_complex = False
                            deleted = True

        self.mark_shape()
    
    def make_shape(self):
        for i in self.edges:
            e = i
            e.on_shape = False
            if(e.triangle.on_complex):
                if(e.inverse_edge == None):
                    e.on_shape = True
                elif(not e.inverse_edge.triangle.on_complex):
                    e.on_shape = True
    
    def edge_is_exposed(self, alpha, e):
        x1 = e.node1.x
        x2 = e.node2.x
        y1 = e.node1.y
        y2 = e.node2.y
        xe = (x1 + x2) / 2
        ye = (y1 + y2) / 2
        d = math.sqrt(alpha**2 -e.weight**2 / 4)
        xt = d * (y2 -y1) / e.weight
        yt = d * (x2 -x1) / e.weight
        xc1 = xe + xt
        yc1 = ye - yt
        xc2 = xe - xt
        yc2 = ye + yt
        points_in_circle1 = self.compute_points_in_circle(e.node1, xc1, yc1, alpha) or self.compute_points_in_circle(e.node2, xc1, yc1, alpha)
        points_in_circle2 = self.compute_points_in_circle(e.node1, xc2, yc2, alpha) or self.compute_points_in_circle(e.node2, xc2, yc2, alpha)

        return not (points_in_circle1 and points_in_circle2)
    
    def compute_stringy_measure(self):
        count1 = 0
        count2 = 0
        for i in self.nodes:
            if(n.mst_degree == 1):
                count1 += 1
            if(n.mst_degree == 2):
                count2 += 2
        
        result = count2/(len(self.nodes) - count1)
        return result**3
    
    def compute_cluster_measure(self):
        it = self.mst_edges
        max_length = [0] * 1
        max_value = 0

        for e in it:
            
            self.clear_visit()
            e.on_mst = False
            runts = e.get_runts(max_length)
            e.on_mst = True
            if(max_length[0] > 0):
                value = runts * (1 - max_length[0] / e.weight)
                if(value > max_value):
                    max_value = value
        return 2 *  max_value / self.totalPeeledCount
    
    def clear_visit(self):
        it = self.nodes
        for n in it:
            n.isVisited = False
    

    def compute_monotonicity_measure(self):
        n = len(self.counts)
        ax = [0.0] * n
        ay = [0.0] * n
        weights = [0.0] * n

        for i in range(n):
            ax[i] = self.px[i]
            ay[i] = self.py[i]
            weights[i] = self.counts[i]
        
        rx = Sorts.rank(ax)
        ry = Sorts.rank(ay)
        s = self.compute_pearson(rx, ry, weights)
        return s**2
    
    def compute_pearson(self, x, y, weights):
        n = len(x)
        xmean = 0
        ymean = 0
        xx = 0
        yy = 0
        xy = 0
        sumwt = 0
        for i in range(n):
            wt = weights[i]
            if(wt > 0 and not self.is_outlier[i]):
                sumwt += wt
                xx += (x[i] - xmean) * wt * (x[i] - xmean)
                yy += (y[i] - ymean) * wt * (y[i] - ymean)
                xy += (x[i] - xmean) * wt * (y[i] - ymean)
                xmean += (x[i] - xmean) * wt / sumwt
                ymean += (y[i] - ymean) * wt / sumwt
        
        xy = xy / xy /math.sqrt(xx * yy)
        return xy
    
    def compute_sparseness_measure(self):
        n = len(self.sorted_original_mst_lengths)
        n90 = (9 * n) / 10
        sparse = math.min(self.sorted_original_mst_lengths[n90] / 1000, 1)
        t = self.total_count / 500
        correction = .7 + .3 / (1 + t**2)
        return correction * sparse
    

    def compute_striation_measure(self):
        num_edges = 0
        it = self.mst_edges
        for i in it:
            e = i
            n1 = e.node1
            n2 = e.node2
            if(n1.mst_degree == 2 and n2.mst_degree==2):
                e1 = self.get_adjacent_mst_edge(n1, e)
                e2 = self.get_adjacent_mst_edge(n2, e)

                if(self.cosine_of_adjacent_edges(e, e1, n1) <  -.7 and self.cosine_of_adjacent_edges(e, e2, n2) < -.7):
                    num_edges += 1
        return num_edges / len(self.mst_edges)
    
    def get_adjacent_mst_edge(self, n, e):
        nt = n.neighbors
        for et in nt:
            if(et.on_mst and not (e == et)):
                return et
        return None
    
    def consine_of_adjacent_edges(self, e1, e2, n):
        v1x = e1.other_node(n).x - n.x
        v1y = e1.other_node(n).y - n.y
        v2x = e2.other_node(n).x - n.x
        v2y = e2.other_node(n).y - n.y

        v1 = math.sqrt(v1x**2 + v1y**2)
        v2 = math.sqrt(v2x**2 + v2y**2)

        v1x = v1x / v1
        v1y = v1y / v1
        v2x = v2x / v2
        v2y = v2y / v2

        return v1x * v2x + v1y * v2y
    
    def compute_convexity_measure(self):
        if(self.hull_area == 0):
            return 1
        else:
            t = self.total_count / 500
            correction = 0.7 + 0.3 / (1 + t**2)
            convexity = self.alpha_area / self.hull_area
            return correction * convexity
    
    def compute_skinny_measure(self):
        if(self.alpha_perimeter > 0):
            return 1 - math.sqrt(4 * math.pi * self.alpha_area) / self.alpha_perimeter
        else:
            return 1
    
    def compute_alpha_area(self):
        area = 0
        tri = self.triangles
        for i in tri:
            t = i
            if(t.on_complex):
                p1 = t.edge.node1
                p2 = t.edge.node2
                p3 = t.edge.next_edge.node2

                area += abs(p1.x * p2.y + p1.y * p3.x + p2.x * p3.y
                        - p3.x * p2.y - p3.y * p1.x - p1.y * p2.x)
        self.alpha_area = area / 2

    def compute_hull_area(self):
        area = 0
        tri = self.triangles
        for i in tri:
            t = i
         
            p1 = t.edge.node1
            p2 = t.edge.node2
            p3 = t.edge.next_edge.node2

            area += abs(p1.x * p2.y + p1.y * p3.x + p2.x * p3.y
                    - p3.x * p2.y - p3.y * p1.x - p1.y * p2.x)

        self.hull_area = area / 2

    def compute_alpha_perimeter(self):
        sum = 0 
        it = self.edges
        for i in it:
            e = i
            if(e.on_shape):
                sum += e.weight
        
        self.alpha_perimeter = sum
    
    def compute_hull_perimiter(self):
        sum = 0
        e = self.hull_start
        while(True):
            sum += e.node1.dist_to_node(e.node2.x, e.node2.y)
            e = e.next_convect_hull_link
            if(e == self.hull_start):
                break
        self.hull_perimeter = sum
    
    def set_neighbors(self):
        it = self.edges

        for i in it:
            e = i
            if(e.is_new_edge(e.node1)):
                e.node1.set_neighbor(e)
            if(e.is_new_edge(e.node2)):
                e.node2.set_neighbor(e)
    
    def insert(self, px, py, count, id):
        eid = 0
        nd = Node(px, py, count, id)
        self.nodes.append(nd)

        if(len(self.nodes) < 3):
            return
        if(len(self.nodes) == 3):
            p1 = self.nodes[0]
            p2 = self.nodes[1]
            p3 = self.nodes[2]

            e1 = Edge(p1, p2)

            if(e1.on_side(p3) == 0):
                self.nodes.remove(nd)
                return
            if(e1.on_side(p3) == -1):
                p1 = self.nodes[0]
                p2 = self.nodes[1]
                e1.update(p1, p2)
            
            e2 = Edge(p2, p3)
            e3 = Edge(p3, p1)

            e1.next_edge = e2
            e2.next_edge = e3
            e3.next_edge = e1
        
            self.hull_start = e1
            self.triangles.append(Triangle(self.edges, e1, e2, e3))
            return

        actE = self.edges[0]

        if(actE.on_side(nd) == -1):
            if(actE.inverse_edge == None):
                eid = -1
            else:
                eid = self.search_edge(actE.inverse_edge, nd)
        else:
            eid = self.search_edge(actE, nd)
        if(eid == 0):
            self.nodes.remove(nd)
            return
        if(eid > 0):
            self.expand_triangle(actE, nd, eid)
        else:
            self.expand_hull(nd)
        

    def expande_triangle(self, e, nd, type):
        e1 = e
        e2 = e1.next_edge
        e3 = e2.next_edge

        p1 = e1.node1
        p2 = e2.node1
        p3 = e3.node1

        if (type == 2):
            e10 = Edge(p1, nd)
            e20 = Edge(p2, nd)
            e30 = Edge(p3, nd)

            e.triangle.remove_edges(self.edges)
            self.triangles.remove(e.inverse_edge)
            e100 = e10.make_symm()
            e200 = e20.make_symm()
            e300 = e30.make_symm()

            self.triangles.append(Triangle(self.edges, e1, e20, e100))
            self.triangles.append(Triangle(self.edges, e2, e30, e200))
            self.triangles.append(Triangle(self.edges, e3, e10, e300))
            self.swap_test(e1)
            self.swap_test(e2)
            self.swap_test(e3)
        else:
            e4 = e1.inverse_edge
            if(e4 == None or e4.triangle==None):
                e30 = Edge(p3, nd)
                e02 = Edge(nd, p2)
                e10 = Edge(p1, nd)
                e03 = e30.make_symm()

                e10.as_index()
                e1.most_left.next_convect_hull_link = e10
                e10.next_convect_hull_link = e02
                e02.next_convect_hull_link = e1.next_convect_hull_link
                self.hull_start = e02

                self.triangles.remove(e1.triangle)
                self.edges.remove(e1)
                self.edges.append(e10)
                self.edges.append(e02)
                self.edges.append(e30)
                self.edges.append(e03)
                self.triangles(Triangle(e2, e30, e02))
                self.triangles(Triangle(e3, e10, e03))
                self.swap_test(e2)
                self.swap_test(e3)
                self.swap_test(e30)
            else:
                e5 = e4.next_edge
                e6 = e5.next_edge
                p4 = e6.node1
                e10 = p1, nd
                e20 = p2, nd
                e30 = p3, nd
                e40 = p4, nd
                self.triangles.remove(e4.triangle)
                e.triangle.removeEdges(self.edges)
                self.triangles.remove(e4.triangle)
                e4.triangle.removeEdges(self.edges)
                e5.as_index()
                e2.as_index()
                self.triangles.append(Triangle(self.edges, e2, e30, e20.make_symm()))
                self.triangles.append(Triangle(self.edges, e3, e10, e30.make_symm()))
                self.triangles.append(Triangle(self.edges, e5, e40, e10.make_symm()))
                self.triangles.append(Triangle(self.edges, e6, e20, e40.make_symm()))
                self.swap_test(e2)
                self.swap_test(e3)
                self.swap_test(e5)
                self.swap_test(e6)
                self.swap_test(e10)
                self.swap_test(e20)
                self.swap_test(e30)
                self.swap_test(e40)
    

    def expand_hull(self, nd):
        e1 = Edge()
        d2 = Edge()
        e3 = None
        enext = Edge()

        e = self.hull_start
        comedge = None
        lastbe = None
        while(True):
            enext = e.next_convect_hull_link
            if(e.on_side(nd) == -1):
                if(lastbe != None):
                    e1.make_symm()
                    e2 = Edge(e.node1, nd)
                    e2 = Edge(nd, e.node2)
                    if(comedge == None):
                        self.hull_start = lastbe
                        lastbe.next_convect_hull_link = e2
                        lastbe = e2
                    else:
                        comedge.make_symm()
                    
                    comedge = e3
                    self.triangles.append(Triangle(self.edges, e1, e2, e3))
                    self.swap_test(e)
            else:
                if(comedge != None):
                    break
                lastbe = e
            e = enext
        
        lastbe.next_convect_hull_link = e3
        e3.next_convect_hull_link = e
    

    def search_edge(self, e, nd):
       
        f2 = e.next_edge.on_side(nd)

        if(f2 == -1):
            if(e.next_edge.inverse_edge != None):
                return self.search_edge(e.next_edge.inverse_edge, nd)
            else:
                self.act_e  = e
                return -1
        if(f2 == 0):
            e0 = e.next_edge
        ee = e.next_edge
        f3 =  ee.next_edge.on_side(nd)
        if(f3 == -1):
            if(ee.next_edge.inverse_edge != None):
                return self.search_edge(ee.next_edge.inverse_edge, nd)
            else:
                self.act_e = ee.next_edge
                return -1
        
        if(f3 == 0):
            e0 = ee.next_edge
        if(e.on_side(nd) == 0):
            e0 = e
        if(e0 != None):
            self.act_e = e0
            if(e0.next_edge.on_side(nd) == 0):
                self.act_e = e0.next_edge
                return 0
            if (e0.next_edge.next_edge.on_side(nd) == 0):
                return 0
            return 1
        self.act_e = ee
        return 2

    def swap_test(self, e11):
        e21 = e11.inverse_edge
        if(e21 == None or e21.triangle == None):
            return
        e12 = e11.next_edge()
        e13 = e12.next_edge()
        e22 = e21.next_edge()
        e23 = e23.next_edge()

        if(e11.trinagle.in_circle(e22.node2) or e21.trinagle.in_circle(e12.node2)):
            e11.update(e22.node2, e12.node2)
            e21.update(e12.node2, e22.node2)
            e11.link_symm()
            e12.triangle.update(e13, e22, e11)
            e12.triangle.update(e23, e12, e21)
            e12.as_index()
            e22.as_index()
            self.swap_test(e12)
            self.swap_test(e22)
            self.swap_test(e13)
            self.swap_test(e23)
    
    def mark_hull(self):
        e = self.hull_start
        if(e != None):
            while(True):
                e.on_hull = True
                e.node1.on_hull = True
                e.node2.on_hull = True
                e = e.next_edge
                if(e == self.hull_start):
                    break





        





                

    








                    



        



