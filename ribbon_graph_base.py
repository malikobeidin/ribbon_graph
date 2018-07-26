from sage.all import PermutationGroup
from permutation import Permutation, Bijection, random_permutation
from cycle import Path, EmbeddedPath, EmbeddedCycle
from spherogram.links.random_links import map_to_link, random_map
from random import choice
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
            opposite, next = self._permutations_from_PD(PD)
        self.opposite = opposite
        self.next = next        
        self.next_corner = self.opposite * self.next.inverse()

        
    def __repr__(self):
        return "RibbonGraph with {} half-edges and {} vertices".format(self.size(), len(self.vertices()))

    def _permutations_from_PD(self, PD):
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
        return opposite,next
    
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

    def euler_characteristic(self):
        return len(self.vertices()) - len(self.edges()) + len(self.faces())
    
    def lace_component(self, label):
        return (self.opposite*self.next*self.next).cycle(label)
    
    def lace_components(self):
        return (self.opposite*self.next*self.next).cycles()
    
    def size(self):
        return len(self.opposite)

    def make_new_labels(self, num_labels):
        max_int_label = 0
        for j in self.labels():
            try:
                j_int = int(j)
                if j_int > max_int_label:
                    max_int_label = j_int
            except (ValueError, TypeError):
                try:
                    j_int = int(max(j))
                    if j_int > max_int_label:
                        max_int_label = j_int
                except:
                    continue
        return range(max_int_label + 1, max_int_label + 1 + num_labels)

        
    def dual(self):
        """
        Return the dual RibbonGraph, i.e. the RibbonGraph where the vertices
        are the faces of the original, and the edges correspond to edges of the
        original.

        The dual of the dual is NOT exactly the original. It is the original,
        but with all of the labels switched with their opposites.
        That is, the 'next' permutation has been conjugated by the 'opposite'
        permutation.
        """
        
        return RibbonGraph(permutations=[self.opposite, self.next_corner])

    def mirror(self):
        """
        Return the mirror image of the RibbonGraph, which has the same edges
        but in the reverse order around each vertex.
        """
        return RibbonGraph(permutations=[self.opposite, self.next.inverse()])
    
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

    def disconnect_vertices(self, labels):
        """
        Given list of half edges, pull off the labels at each vertex.
        """
        label_set = set(labels)
        new_next = self.next
        for label in label_set:
            switch_perm = Permutation({label:new_next[label],new_next[label]:label })
            new_next = new_next * switch_perm
        return RibbonGraph([self.opposite, new_next])

    def delete_edges(self, labels):
        """
        Given a list of labels, delete the entire edge that each label in the
        list in on.
        """
        labels = set(labels)
        labels_with_opposite_labels = set()
        for label in labels:
            for other_label in self.edge(label):
                labels_with_opposite_labels.add(other_label)
        return self.delete_labels(labels_with_opposite_labels)

        
    def delete_vertices(self, labels):
        """
        Given a list of labels, delete the entire vertex that each label in the
        list in on.
        """
        labels = set(labels)
        labels_with_next_labels = set()
        for label in labels:
            for other_label in self.vertex(label):                
                labels_with_next_labels.add(other_label)
        return self.delete_labels(labels_with_next_labels)
        
    
    def delete_labels(self, labels):
        return self.disconnect_edges(labels).disconnect_vertices(labels).remove_labels(labels)
    
    def connect_edges(self, pairing):
        """
        Given a list of pairs of half-edge labels which are currently 
        disconnected (fixed points of self.opposite), connect the half-edges up.
        If one of the labels is already connected, it raises an exception.
        """
        connecting_permutation = Permutation(cycles=pairing)
        all_labels = connecting_permutation.labels()
        for label in connecting_permutation.labels():
            if self.opposite[label] != label:
                raise Exception("Trying to connect already connected half edge")
        new_op = self.opposite.restricted_to(self.opposite.labels()-all_labels)
        return RibbonGraph([new_op*connecting_permutation, self.next])

    
    def disjoint_union(self, other_ribbon_graph):
        new_op = self.opposite.disjoint_union(other_ribbon_graph.opposite)
        new_next = self.next.disjoint_union(other_ribbon_graph.next)
        return RibbonGraph([new_op, new_next])
        
    
    def glue_along_vertex(self, label, other_ribbon_graph, other_label):
        vertex = self.vertex(label)
        other_vertex = other_ribbon_graph.vertex(other_label)
        if len(vertex) != len(other_vertex):
            raise Exception("Must glue along two vertices of same size")        
        
        
    def union(self, other_ribbon_graph):
        """
        Combine self and other_ribbon_graph, without forcing the labels to be
        different as in disjoint_union.
        """
        new_opposite = self.opposite.union(other_ribbon_graph.opposite)
        new_next = self.next.union(other_ribbon_graph.next)
        
        return RibbonGraph([new_opposite, new_next])
    
        
    def glue_along_face(self, label, other_ribbon_graph, other_label):
        """
        Given two embedded faces (no vertex is encountered twice when walking
        around boundary) of the same size on two different ribbon graphs,
        glue along that boundary cycle so that label in self ends up
        connected to (an opposite) other_label in other_ribbon_graph.
        """
        
        face_length = len(self.face(label))
        if len(other_ribbon_graph.face(other_label)) != face_length:
            raise Exception("Faces must have same size")
        cycle = EmbeddedCycle(self, label, turn_degrees = [-1]*face_length)
        other_cycle = EmbeddedCycle(other_ribbon_graph, other_label, turn_degrees = [-1]*face_length).reversed()
        union_ribbon_graph = self.disjoint_union(other_ribbon_graph)
        for l, ol in zip(cycle.labels[:-1], other_cycle.labels[:-1]):
            nl, nol = self.next[l], other_ribbon_graph.next[ol]
            #the following function glues two vertices in the slots BEFORE
            #the given labels. So, need to shift them by one first, as above.
            union_ribbon_graph = union_ribbon_graph.vertex_merge_unmerge((nl,0),(nol,1) )
        #Edges on mutual boundary are now doubled, need to cut half of them.
        #we cut the ones from other_ribbon_graph, as a convention.
        doubled_edges = []
        for ol in other_cycle.labels[:-1]:
            doubled_edges.append((ol,1))
            doubled_edges.append((other_ribbon_graph.opposite[ol],1))

        union_ribbon_graph = union_ribbon_graph.delete_labels(doubled_edges)
        return union_ribbon_graph

    
    def add_new_vertices(self, vertex_labels, vertex_size):
        """
        Add a new vertex for each label in vertex_labels, with each vertex
        having size vertex_size. The half edges around the vertex 
        corresponding to l will be labeled (l,0),(l,1),...(l,vertex_size-1).
        They are not yet connected to any other vertices.
        """

        new_op = self.opposite * Permutation({(l,i):(l,i) for l in vertex_labels for i in range(vertex_size)} )
        new_next = self.next * Permutation(cycles = [[(l,i) for i in range(vertex_size)] for l in vertex_labels]) 

        return RibbonGraph([new_op, new_next])
    
    def connect_vertices(self, pairing):
        pass
    
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
        return RibbonGraph(permutations = [self.opposite, self.next*Permutation({a:b,b:a})])
    
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
        
    def draw_strand_along_path(self, start, end, path):
        """
        Given an EmbeddedPath in the DUAL ribbon graph
        from two disconnected labels (i.e., labels 'start' and 'end'
        for which self.opposite is a fixed point), create a strand from one
        to the other crossing along the edges in path.
        The label 'start' must be on the same face as the first label in path,
        and the label 'end' must be OPPOSITE a label on the same face as the
        last label in path. So, path is the sequence of edges you must cross to
        get from the face containing start to the face containing end.
        """
        op = self.opposite
        next = self.next
        if  op[start] != start:
            raise Exception("The start label is already connected.")
        if op[end] != end:
            raise Exception("The end label is already connected.")
        if path.labels[0] not in self.face(start):
            raise Exception("Path must begin on same face as start label.")
        if self.opposite[path.labels[-1]] not in self.face(end):
            raise Exception("Path must finish on a label opposite the face of the end label.")
        
        new_ribbon_graph = self.disconnect_edges(path.labels)
        new_labels = new_ribbon_graph.make_new_labels(len(path.labels))
        new_ribbon_graph = new_ribbon_graph.add_new_vertices(new_labels, 4)

        #Making the connections along the path itself
        connections = [(start, (new_labels[0],0))]
        for i in range(len(new_labels)-1):
            new_label, next_new_label = new_labels[i], new_labels[i+1]
            connections.append( ( (new_label,2) , (next_new_label,0) ) )
        connections.append((end, (new_labels[-1], 2)))

        #Making the connections on either side of the path (to the edges crossed)
        for new_label, label in zip(new_labels, path.labels):
            connections.append((label,(new_label,1)))
            connections.append((self.opposite[label],(new_label,3)))


        return new_ribbon_graph.connect_edges(connections)

    def move_strand_off_crossing(self, label, ccw=True):
        """
        Take the strand defined by label and move it around the vertex in the
        counterclockwise direction (unless ccw is set to False).
        If the number of labels around the vertex is odd, then there are no
        'strands' so it raises an exception.
        """
        vertex = self.vertex(label)
        
    
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

    def search_for_embedded_cycles(self, max_turn, length):
        cycle_types = self.search_for_cycles(max_turn, length)
        cycles = []
        for p in cycle_types:
            pp = self.path_permutation(p)
            fixed_points = pp.fixed_points()
            if len(fixed_points) < self.size():
                while fixed_points:
                    start_point = fixed_points.pop()
                    cycle = None
                    try:
                        cycle = EmbeddedCycle(self, start_point, turn_degrees=p)
                    except:
                        cycle = Path(self, start_point, turn_degrees=p)
                    fixed_points = fixed_points - set(cycle.labels)
                    if isinstance(cycle, EmbeddedCycle):
                        cycles.append(cycle)
        return cycles

    def search_for_embedded_cycles_through(self, start_point, length):
        embedded_paths = [EmbeddedPath(self, start_point, labels = [start_point])]
        for i in range(length-1):
            new_paths = []
            for path in embedded_paths:
                new_paths.extend(path.one_step_continuations())

            embedded_paths = new_paths
        return [P.complete_to_cycle() for P in embedded_paths if P.is_completable_to_cycle()]

    def search_for_long_embedded_cycles_through(self, start_point, max_length):
        embedded_paths = [EmbeddedPath(self, start_point, labels = [start_point])]
        max_length_cycles = []
        for i in range(max_length-1):
            new_paths = []
            for path in embedded_paths:
                new_paths.extend(path.one_step_continuations())

            if new_paths:
                embedded_paths = new_paths
                cycles = [P.complete_to_cycle() for P in embedded_paths if P.is_completable_to_cycle()]
                if cycles:
                    max_length_cycles = cycles
            else:
                break
        return max_length_cycles

    def search_for_embedded_cycle_with_start_and_goal(self, start, goal_labels, max_length):
        embedded_paths = [EmbeddedPath(self, start, labels = [start])]
        cycles_through_goal = []
        for i in range(max_length-1):
            new_paths = []
            for path in embedded_paths:
                new_paths.extend(path.one_step_continuations())

            if new_paths:
                embedded_paths = new_paths
                cycles = [P.complete_to_cycle() for P in embedded_paths if P.is_completable_to_cycle()]
                for cycle in cycles:
                    for label in cycle.labels:
                        if label in goal_labels:
                            cycles_through_goal.append(cycle)
                            break
            else:
                break
        return cycles_through_goal

        
    
    def copy(self):
        return RibbonGraph([Permutation(self.opposite), Permutation(self.next)])


    def info(self):
        print("Vertices: {}\nEdges: {}\nFaces: {}".format(self.vertices(), self.edges(), self.faces()))

        
    def random_label(self):
        return choice(tuple(self.labels()))

    def sage(self):
        from sage.all import Graph
        G = Graph(multiedges=True, loops=True)

        vertices = map(tuple,self.vertices())
        edges = map(tuple, self.edges())
        embedding = {}
        for vertex in vertices:
            vertex_order = []
            for label in vertex:
                for edge in edges:
                    if label in edge:
                        break
                G.add_edge(vertex,edge)
                vertex_order.append(edge)
            embedding[vertex]=vertex_order

        for edge in edges:
            edge_order = []
            for label in edge:
                for vertex in vertices:
                    if label in vertex:
                        break
                edge_order.append(vertex)
            embedding[edge]=edge_order
            
        G.set_embedding(embedding)
        return G

    
def random_link_shadow(size, edge_conn=2):
    PD = map_to_link(random_map(size, edge_conn_param=edge_conn)).PD_code()
    return RibbonGraph(PD=PD)

def trefoil():
    next = Permutation(cycles=[('a','b','c','d'),
                               ('e','f','g','h'),
                               ('i','j','k','l')])
    opposite = Permutation(cycles=[('a','f'),
                                ('b','e'),
                                ('c','l'),
                                ('d','k'),
                                ('i','h'),
                                ('g','j')])
    return RibbonGraph([opposite, next])

def loops():
    next = Permutation(cycles=[(1,2,3,4),
                               (5,6,7,8)])
    opposite = Permutation(cycles=[(1,5),
                                (2,8),
                                (3,4),
                                (6,7)])
    return RibbonGraph([opposite, next])

def cube():
    next = Permutation(cycles=[(1,2,3),
                               (4,5,6),
                               (7,8,9),
                               (10,11,12),
                               (13,14,15),
                               (16,17,18),
                               (19,20,21),
                               (22,23,24)])
    opposite = Permutation(cycles=[(1,5),
                                   (4,8),
                                   (7,11),
                                   (2,10),
                                   (13,18),
                                   (16,21),
                                   (19,24),
                                   (22,15),
                                   (3,14),
                                   (6,17),
                                   (9,20),
                                   (12,23)])
    return RibbonGraph([opposite, next])
