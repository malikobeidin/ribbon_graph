import matplotlib.pyplot as plt
import numpy as np
from cycle import EmbeddedCycle
from decompositions import CycleTree

def draw_with_plink(ribbon_graph):
    pass

class Immersion(object):
    """

    """
    def __init__(self, ribbon_graph, head, tail):
        self.ribbon_graph = ribbon_graph
        self.head = head
        self.tail = tail

    def energy(self):
        pass
    
    def perturb_downward(self, step):
        pass
    
    def perturb_randomly(self, step):
        pass

    def minimize_energy(self):
        pass
    
    def draw(self):
        pass

class TutteSpringEmbedding(object):
    def __init__(self, ribbon_graph, outer_face_label):
        outer_vertices = [frozenset(ribbon_graph.vertex(l)) for l in ribbon_graph.face(outer_face_label)]
        all_vertices = [frozenset(v) for v in ribbon_graph.vertices()]

        outer_vertex_indices = [all_vertices.index(v) for v in outer_vertices]

        print(all_vertices)
        print(outer_vertices)
        print(outer_vertex_indices)
        self.outer_vertex_indices = outer_vertex_indices
        adjacencies = []
        for v in all_vertices:
            opposites = [ribbon_graph.opposite[l] for l in v]
            if v not in outer_vertices:
                adjacencies.append([all_vertices.index(frozenset(ribbon_graph.vertex(l))) for l in opposites])
            else:
                adjacencies.append([])

        print(adjacencies)
        self.adjacencies = adjacencies
        n = len(all_vertices)
        spring_system = np.zeros((n,n))
        for i in range(n):
            spring_system[i][i] = 1
            for j in adjacencies[i]:
                spring_system[i,j] = -1.0/len(adjacencies[i])
                
        print(spring_system)
        self.spring_system = spring_system
        ts = np.linspace(0, 2*np.pi, len(outer_vertices), endpoint = False)
        bx = np.zeros(n)
        for circle_index,vertex_index in enumerate(outer_vertex_indices):
            bx[vertex_index] = np.cos( ts[circle_index] )
        by = np.zeros(n)
        for circle_index,vertex_index in enumerate(outer_vertex_indices):
            by[vertex_index] = -np.sin( ts[circle_index] )

        print(bx)
        print(by)

        self.bx = bx
        self.by = by
        
        xs = np.linalg.solve(spring_system, bx)
        ys = np.linalg.solve(spring_system, by)
        self.xs = xs
        self.ys = ys


    def plot(self, filename):
        xs, ys = self.xs, self.ys
        plt.scatter(xs,ys)
        for v, v_adjacencies in enumerate(self.adjacencies):
            for w in v_adjacencies:
                edge_x = [xs[v], xs[w]]
                edge_y = [ys[v], ys[w]]
                plt.plot(edge_x, edge_y, color='blue')

        outer_vertex_indices = self.outer_vertex_indices
        for i in range(len(outer_vertex_indices)):
            v, next_v = outer_vertex_indices[i], outer_vertex_indices[(i+1)%len(outer_vertex_indices)]
            edge_x = [xs[v], xs[next_v]]
            edge_y = [ys[v], ys[next_v]]
            plt.plot(edge_x, edge_y, color='blue')

        plt.savefig(filename+'.png')


class PolygonDrawing(object):
    def __init__(self, vertices):
        self.vertices = vertices
        self.n = len(vertices)

        self.min_x = min(x for x,y in vertices)
        self.min_y = min(y for x,y in vertices)
        self.max_x = max(x for x,y in vertices)
        self.max_y = max(y for x,y in vertices)
    
    def barycentric_coordinates(self, p):
        weights = np.zeros(self.n)
        for i, vertex in enumerate(self.vertices):
            previous_vertex = self.vertices[(i-1)%self.n]
            next_vertex = self.vertices[(i+1)%self.n]
            prev_cot = self._cotangent(p, vertex, previous_vertex)
            next_cot = self._cotangent(p, vertex, next_vertex)
            dist = np.sum((p-vertex)*(p-vertex))
            weights[i] = (prev_cot + next_cot) / dist
        weights = weights/sum(weights)
        return weights

    def diagonal_linear_system(self, i, j):
        pass

    def trace_diagonal(self, i, j, stepsize, distance_goal=.01):
        """
        Start at vertex i and step in the diagonal direction until you reach
        vertex j.
        """
        pass
        
    
    def diagonal(self, i, j, epsilon = .001, sample_points = 100):
        diagonal_points = []
        for x in np.linspace(self.min_x, self.max_x, sample_points):
            for y in np.linspace(self.min_y, self.max_y, sample_points):
                weights = self.barycentric_coordinates( np.array([x,y]) )
                is_diagonal = True
                for k, w in enumerate(weights):
                    if k not in [i,j]:
                        if w > epsilon:
                            is_diagonal = False
                            break
                if is_diagonal:
                    diagonal_points.append( np.array([x,y]) )
        return diagonal_points
                        
        
    
    def triangulate(self):
        pass
    
    def barycentric_subdivision(self, triangulation):
        pass

    def parametrization(self, num_subdivisions):
        pass
    
    def _cotangent(self, a, b, c):
        ba = a-b
        bc = c-b
        dot = sum(bc*ba)
        det = np.linalg.det(np.array([bc,ba]))
        return dot/abs(det)
        
class GluedPolygonalDrawing(object):
    def __init__(self, ribbon_graph, exterior_label, max_length):
        exterior_face = ribbon_graph.face(exterior_label)
        cycle = EmbeddedCycle(ribbon_graph, exterior_label, turn_degrees=[-1]*len(exterior_face)).reversed()
        self.cycle_tree = CycleTree(cycle, max_length)

    
