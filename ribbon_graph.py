from sage.all import PermutationGroup
from permutation import Permutation
class RibbonGraph():
    def __init__(self, opposite, next, PD = []):
        if PD:
            edge_dict = {}
            for n,v in enumerate(PD):
                for m,label in enumerate(v):                    
                    if label in edge_dict:
                        edge_dict[label].append((n,m))
                    else:
                        edge_dict[label] = [(n,m)]
            positions = []
            for l in edge_dict.values():
                positions.extend(l)

            opposite_list = [[positions.index(pair)+1 for pair in edge_dict[label]] for label in edge_dict ]

            next_list = []
            for n,v in enumerate(PD):
                cycle = []
                for m,label in enumerate(v):
                    cycle.append(positions.index((n,m))+1)
                next_list.append(cycle)

            opposite = Permutation(dictionary={},cycles = opposite_list)
            next = Permutation(dictionary={},cycles = next_list)

        else:
            pass
        assert len(opposite) % 2 == 0
        assert len(next) == len(opposite)
        self.opposite = opposite
        self.next = next
        self.next_corner = self.opposite * self.next

    def vertices(self):
        return self.next.cycles()

    def edges(self):
        return self.opposite.cycles()

    def faces(self):
        return self.next_corner.cycles()

    def lace_components(self):
        return (self.opposite*self.next*self.next).cycles()
    
    def size(self):
        return len(self.opposite)

    def dual(self):
        return RibbonGraph(self.opposite, self.next_corner)

    def labels(self):
        return self.opposite.labels()

    def relabeled(self):
        labels = list(self.labels())
        indices = {l:i for i,l in enumerate(labels)}
        new_op = Permutation({i:indices[self.opposite[labels[i]]] for i in range(len(labels))})
        new_next = Permutation({i:indices[self.next[labels[i]]] for i in range(len(labels))})
        return RibbonGraph(new_op,new_next)
    
    def vertex_merge_unmerge(self,a,b):
        """
        If a and b are on the same vertex, disconnects the vertex at the 
        corners before a and b.
        If a and b are on different vertices, connects the two vertices at
        the corners before a and b.
        If a or b are not in the set of labels already present, this adds
        a new dart
        """
        return RibbonGraph(self.opposite, self.next*Permutation({a:b,b:a}))
    
    def orientations(self):
        vertices = self.vertices()
        n = self.next
        o = self.opposite
        orientations = {}
        p = n
        pi = n.inverse()
        while vertices:
            p = p*o*n*n
            pi = pi*o*n*n
            for i in p.fixed_points():
                for vertex in vertices:
                    if i in vertex:
                        orientations[vertex]=1
                        vertices.remove(vertex)
            for i in pi.fixed_points():
                for vertex in vertices:
                    if i in vertex:
                        orientations[vertex]=-1
                        vertices.remove(vertex)

        return orientations

    def permutation_subgroup(self):
        
        op = map(tuple, self.opposite.cycles())
        next = map(tuple, self.next.cycles())
        return PermutationGroup([op,next])

    def medial_map(self):
        next_corner = self.next_corner
        next_corner_inverse = self.next_corner.inverse()
        labels = self.labels()
        new_next_dict = {}
        for i in labels:
            j = self.opposite[i]
            new_next_dict[(i,1)] = (j, -1)
            new_next_dict[(j,-1)] = (j, 1)
            new_next_dict[(j,1)] = (i, -1)
            new_next_dict[(i,-1)] = (i, 1)
        new_next = Permutation(new_next_dict)
        new_op_dict = {}
        for i in labels:
            for s in [-1,1]:
                new_op_dict[(i,1)] = (next_corner[i],-1)
                new_op_dict[(i,-1)] = (next_corner_inverse[i],1)
        new_op = Permutation(new_op_dict)

        return RibbonGraph(new_op, new_next)
        
    
    def PD_code(self):
        vertices = self.vertices()
        edges = self.edges()
        pd = []
        for v in vertices:
            vertex_code = []
            for i in v:
                for j, edge in enumerate(edges):
                    if i in edge:
                        vertex_code.append(j)
                        break
            pd.append(vertex_code)
        return pd
