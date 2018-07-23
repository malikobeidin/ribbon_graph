import matplotlib.pyplot as plt
import numpy as np

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
