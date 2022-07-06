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
