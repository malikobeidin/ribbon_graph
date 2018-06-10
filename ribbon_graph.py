from sage.all import PermutationGroup
from permutation import Permutation, Bijection, random_permutation
from spherogram.links.random_links import map_to_link, random_map
import itertools

class RibbonGraph(object):
    """
    A RibbonGraph consists of a pair of permutations on a set of labels.
    Each label corresponds to a half-edge, and the permutations 'opposite' and
    'next' determine how the half-edges are connected to one another.
    'opposite' determines which half-edge is on the opposite side of the same 
    edge. 'next' determines which half-edge is the next counterclockwise 
    half-edge on the same vertex.

    The permutation next_corner is simply the product of opposite and next, and
    is used frequently, so it is computed when the object is instantiated. It is
    used to determine the faces of a RibbonGraph.
    """
    
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
        return "RibbonGraph with {} half-edges and {} vertices".format(self.size(), len(self.vertices()))
        
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

    def connected_component(self, label):
        """
        Return all labels in the connected component of label. That is, all
        labels which can be reached by applying the permutations opposite and 
        next.
        """
        
        verts = self._vertex_search(label)
        return set([l for v in verts for l in self.vertex(v)])
        
    def connected_components(self):
        """
        Return all connected components.
        """
        labels = self.next.labels()
        conn_comps = []
        while labels:
            label = labels.pop()
            comp = self.connected_component(label)
            labels = labels-comp
            conn_comps.append(comp)
        return conn_comps

    def restricted_to_connected_component_containing(self, label):
        """
        Return a RibbonGraph (not just a set of labels) corresponding to the
        connected component containing label.
        """
        comp = self.connected_component(label)
        new_op = self.opposite.restricted_to(comp)
        new_next = self.next.restricted_to(comp)
        return RibbonGraph([new_op, new_next])
    
    def _relabeling_bijection(self, label):
        i = 1
        bij = {}
        for oriented_edge in self._vertex_search(label):
            for e in self.vertex(oriented_edge):
                bij[e]=i
                i += 1
        return Bijection(bij)

    def relabeled_by_root(self, label):
        """
        Change the labels so that they are ordered in a canonical way from 
        RibbonGraph._vertex_search starting at label.
        """
        
        bij = self._relabeling_bijection(label)
        new_op = self.opposite.relabeled(bij)
        new_next = self.next.relabeled(bij)
        return RibbonGraph([new_op, new_next])

    def rooted_isomorphism_signature(self, label):
        """
        Returns a list of information which determines the RibbonGraph up to 
        rooted isomorphism with root label. That is to say, if another 
        RibbonGraph and one if its edges returns the same list of information,
        the two RibbonGraphs are isomorphic to each other in such a way that 
        the roots correspond to one another as well.
        """
        edges = self.relabeled_by_root(label).edges()
        return sorted([sorted(e) for e in edges])
    
    def isomorphism_signature(self):
        """
        Return a list of information which determines the isomorphism type 
        of the RibbonGraph.
        """
        return min(self.rooted_isomorphism_signature(label) for label in self.labels())

    def vertex(self, label):
        """
        The vertex containing label
        """
        return self.next.cycle(label)
        
    def vertices(self):
        return self.next.cycles()

    def edge(self, label):
        """
        The edge containing label
        """
        return self.opposite.cycle(label)

    def edges(self):
        return self.opposite.cycles()

    def face(self, label):
        """
        The face containing label.
        """
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
        """
        Return the dual RibbonGraph, i.e. the RibbonGraph where the vertices
        are the faces of the original, and the edges correspond to edges of the
        original.
        """
        
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
        Given list of half edges, disconnect the corresponding edges.
        """
        opposite_labels = set(self.opposite[label] for label in labels)
        all_labels = set(labels).union(opposite_labels)
        new_op = self.opposite.restricted_to(self.opposite.labels()-all_labels)
        for label in all_labels:
            new_op[label] = label
        return RibbonGraph([new_op, self.next])

    
    def connect_edges(self, pairing):
        """
        Given a list of pairs of half-edge labels which are currently 
        disconnected (fixed points of self.opposite), connect the half-edges up.
        If one of the labels is already connected, it raises and exception.
        """
        connecting_permutation = Permutation(cycles=pairing)
        all_labels = connecting_permutation.labels()
        for label in connecting_permutation.labels():
            if self.opposite[label] != label:
                raise Exception("Trying to connect already connect half edge")
        new_op = self.opposite.restricted_to(self.opposite.labels()-all_labels)
        return RibbonGraph([new_op*connecting_permutation, self.next])
    
    def remove_labels(self, labels):
        old_op_labels = self.opposite.labels()
        new_op = self.opposite.restricted_to(old_op_labels-set(labels))

        old_next_labels = self.next.labels()
        new_next = self.next.restricted_to(old_next_labels-set(labels))

        return RibbonGraph([new_op, new_next])
    
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

    def copy(self):
        return RibbonGraph([Permutation(self.opposite), Permutation(self.next)])

    
    
class EmbeddedCycle(object):
    def __init__(self, ribbon_graph, start_label, labels = [], turn_degrees = [], label_set = set([])):
        self.ribbon_graph = ribbon_graph
        self.start_label = start_label

        if labels:
            self.turn_degrees = self._compute_turn_degrees_from_labels(labels)
            self.labels = labels
            
        elif turn_degrees:
            self.turn_degrees = turn_degrees
            self.labels = self._compute_labels_from_turn_degrees(turn_degrees)
                        
        elif label_set:
            self.labels = self._compute_labels_from_label_set(label_set)
            self.turn_degrees = self._compute_turn_degrees_from_labels(self.labels)
        else:
            raise Exception("Must specify labels in cycle, the turn degrees, or the set of labels in the cycle")

        self._verify_embedded()

    def _compute_labels_from_turn_degrees(self, turn_degrees):
        labels = [self.start_label]
        label = self.start_label
        for d in turn_degrees:
            label = self.ribbon_graph.opposite[label]
            label = self.ribbon_graph.next.iterate(d, label)
            labels.append(label)
        if labels[0] == labels[-1]:            
            return labels[:-1]
        else:
            raise Exception("Path is not closed")

    def _compute_turn_degrees_from_labels(self, labels):
        turn_degrees = []
        for i, label in enumerate(labels):
            next_label = labels[(i+1)%len(labels)]
            op_label = self.ribbon_graph.opposite[label]
            vertex = self.ribbon_graph.vertex(op_label)
            turn_degrees.append(vertex.index(next_label))
        return turn_degrees

    def _compute_labels_from_label_set(self, label_set):
        labels = []
        label = self.start_label
        while label_set:
            labels.append(label)
            label = self.ribbon_graph.opposite[label]
            vertex = self.ribbon_graph.vertex(label)
            possible_next_labels = [l for l in label_set if l in vertex]
            if len(possible_next_labels) != 1:
                raise Exception("Label set does not define unique cycle")
            label = possible_next_labels[0]
            label_set.remove(label)
        return labels
            

    def _verify_embedded(self):
        for label in self.labels:
            vertex = self.ribbon_graph.vertex(label)
            for other_label in vertex:
                if (other_label != label) and (other_label in self.labels):
                    raise Exception("Cycle is not embedded")
            

    def __repr__(self):
        return "EmbeddedCycle({})".format(self.labels)
    
    
def random_link_shadow(size, edge_conn=2):
    PD = map_to_link(random_map(size, edge_conn_param=edge_conn)).PD_code()
    return RibbonGraph(PD=PD)
