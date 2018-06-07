from sage.all import PermutationGroup
from permutation import Permutation, Bijection, random_permutation
from spherogram.links.random_links import map_to_link, random_map
import itertools

class RibbonGraph(object):
    def __init__(self, permutations=[], PD = []):
        if permutations:
            opposite, next = permutations
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

        self.opposite = opposite
        self.next = next        
        self.next_corner = self.opposite * self.next

        
    def __repr__(self):
        return "RibbonGraph with {} half-edges".format(self.size())
        
    def _vertex_search(self, label):
        """
        Starting with an oriented half edge label, perform a breadth first
        search of all the vertices to give a canonical ordering of the vertices
        and a choice of oriented half edge for each vertex
        """
        all_seen_edges = set()
        first_edges = []
        stack = []
        stack.append(label)
        num_labels = self.size()
        while stack and len(all_seen_edges) < num_labels:
            oriented_edge = stack.pop()
            if oriented_edge not in all_seen_edges:
                first_edges.append(oriented_edge)
                
                for label in self.vertex(oriented_edge):
                    all_seen_edges.add(label)

                    stack.append(self.opposite[label])

        return first_edges

    def _relabeling_bijection(self, label):
        i = 1
        bij = {}
        for oriented_edge in self._vertex_search(label):
            for e in self.vertex(oriented_edge):
                bij[e]=i
                i += 1
        return Bijection(bij)

    def relabeled_by_root(self, label):
        bij = self._relabeling_bijection(label)
        new_op = self.opposite.relabeled(bij)
        new_next = self.next.relabeled(bij)
        return RibbonGraph([new_op, new_next])

    def rooted_isomorphism_signature(self, label):
        edges = self.relabeled_by_root(label).edges()
        return sorted([sorted(e) for e in edges])
    
    def isomorphism_signature(self):
        return min(self.rooted_isomorphism_signature(label) for label in self.labels())

    def vertex(self, label):
        return self.next.cycle(label)
        
    def vertices(self):
        return self.next.cycles()

    def edge(self, label):
        return self.opposite.cycle(label)

    def edges(self):
        return self.opposite.cycles()

    def face(self, label):
        return self.next_corner.cycle(label)
    
    def faces(self):
        return self.next_corner.cycles()

    def lace_component(self, label):
        return (self.opposite*self.next*self.next).cycle(label)
    
    def lace_components(self):
        return (self.opposite*self.next*self.next).cycles()
    
    def size(self):
        return len(self.opposite)

    def dual(self):
        return RibbonGraph(permutations=[self.opposite, self.next_corner])

    def labels(self):
        return self.opposite.labels()

    def with_shuffled_labels(self):
        bijection = random_permutation(self.labels())
        new_opposite = self.opposite.relabeled(bijection)
        new_next = self.next.relabeled(bijection)
        return RibbonGraph(permutations=[new_opposite, new_next])
    
    def relabeled(self):
        labels = list(self.labels())
        indices = {l:i for i,l in enumerate(labels)}
        new_op = Permutation({i:indices[self.opposite[labels[i]]] for i in range(len(labels))})
        new_next = Permutation({i:indices[self.next[labels[i]]] for i in range(len(labels))})
        return RibbonGraph(new_op,new_next)

    def disconnect_edges(self, labels):
        """
        Given list of half edges, where we choose exactly one from each edge,
        disconnect each of those edges.
        """

        for label in labels:
            assert self.opposite[label] not in labels
            assert label in self.opposite

        all_labels = labels[:]
        all_labels.extend([self.opposite[label] for label in labels])
        disconnecting_perm = Permutation(self.opposite.restricted_to(all_labels))
        return RibbonGraph([self.opposite*disconnecting_perm, self.next])

    
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

        return RibbonGraph(permutations=[new_op, new_next])
        
    
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


    def path_permutation(self, cycle_type):
        perm = self.opposite
        for turn_amount in cycle_type:
            for i in range(turn_amount):
                perm = perm * self.next
            perm = perm*self.opposite
        perm = perm*self.opposite
        return perm

    def search_for_cycles(self,max_turn, length):
        cycle_types = []
        for cycle_type in itertools.product(*[range(1,max_turn+1) for i in range(length)]):
            perm = self.path_permutation(cycle_type)
            if perm.fixed_points():
                cycle_types.append(cycle_type)
        return cycle_types

def random_link_shadow(size, edge_conn=2):
    PD = map_to_link(random_map(size, edge_conn_param=edge_conn)).PD_code()
    return RibbonGraph(PD=PD)
