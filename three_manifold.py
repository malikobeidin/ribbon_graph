from permutation import Permutation, Bijection
from ribbon_graph_base import *

class Triangulation(object):
    def __init__(self, tetrahedra_ribbon_graph, glued_to, check_consistency = True):
        """
        Specify a triangulation of a 3-manifold with:
               tetrahedra_ribbon_graph: a ribbon graph whose faces are all 
                      size 3, and whose connected components are all
                      the tetrahedral map.
               glued_to: a permutation which shows which faces are glued. 
                      A directed edge in a tetrahedron uniquely specifies
                      how to glue two faces, by gluing the faces to the left
                      of each directed edge. This permutation must commute with
                      the next_corner permutation of the tetrahedra_ribbon_graph
                      to make it so that faces are glued consistently.
        """
        self.tetrahedra_ribbon_graph = tetrahedra_ribbon_graph
        if check_consistency:            
            next_corner = self.tetrahedra_ribbon_graph.next_corner()
            glued_to = next_corner.make_commute_along_cycles(glued_to)
        self.glued_to = glued_to

    def tetrahedra(self):
        return self.tetrahedra_ribbon_graph.connected_components()

    def edges(self):
        return (self.tetrahedra_ribbon_graph.next * D.glued_to).cycles()


        
    
    def vertex_links(self):
        return RibbonGraph([self.glued_to, self.tetrahedra_ribbon_graph.next.inverse()])

class TruncatedTetrahedron(object):
    def __init__(self, tet):
        self.tetrahedron = tet
        self.ribbon_graph = truncate_vertices(tetrahedron())
        self.pairing = self.snappy_tetrahedron_to_edge_pairing()
        self.labeled = self.labeled_ribbon_graph()
        
    def vertex(self, directed_edge):
        """
        From a directed edge 'ab', get the vertex in the truncated tetrahedron
        which has this vertex. Each directed edge in the original tetrahedron
        corresponds to a vertex in the truncated tetrahedron.
        """
        return self.ribbon_graph.vertex(directed_edge)
    
    def boundary_edge_from_directed_edge(self, directed_edge):
        """
        Return a label on the triangular face on which the vertex corresponding
        to directed_edge is located.
        """
        return self.ribbon_graph.next[directed_edge]

    def snappy_tetrahedron_to_edge_pairing(self):
        tet = self.tetrahedron
        gluing = tet.Gluing
        neighbor = tet.Neighbor

        directed_edge_choices = ['01','10','23','32']        
        corresponding_left_faces = [7, 11, 13, 14]

        permutations = [Permutation(gluing[i].dict) for i in corresponding_left_faces]
        corresponding_neighbors = [neighbor[i].Index for i in corresponding_left_faces]
        pairing = []
        for e, perm, index in zip(directed_edge_choices,permutations, corresponding_neighbors):
            vs = self.boundary_edge_from_directed_edge(e)
            new_vs = [str(perm[int(v)]) for v in vs]
            new_vs = ''.join([new_vs[2], new_vs[3], new_vs[0], new_vs[1]])
            pairing.append( (str(tet.Index)+vs, str(index)+new_vs) )
        return pairing

    def labeled_ribbon_graph(self):
        l = str(self.tetrahedron.Index)
        new_opposite = {(l+i):(l+self.ribbon_graph.opposite[i]) for i in self.ribbon_graph.opposite}
        new_next = {(l+i):(l+self.ribbon_graph.next[i]) for i in self.ribbon_graph.next}
        
        return RibbonGraph([Permutation(new_opposite), Permutation(new_next)])

def heegaard_surface(mcomplex):
    truncated_tets = [TruncatedTetrahedron(tet) for tet in mcomplex.Tetrahedra]
    truncated_tet = truncated_tets.pop()
    R = truncated_tet.labeled
    pairings = [truncated_tet.pairing]
    for truncated_tet in truncated_tets:
        R = R.union(truncated_tet.labeled)
        pairings.append(truncated_tet.pairing)
    print(pairings)
    
    for pairing in pairings:
        for label1, label2 in pairing:
            labels = R.labels()
            if (label1 in labels) and (label2 in labels):            
                R = R.glue_faces(label1, label2)
    return R
    


N   = 0  # 0000
V0  = 1  # 0001
V1  = 2  # 0010
E01 = 3  # 0011 <-----|
V2  = 4  # 0100       |
E02 = 5  # 0101 <---| |
E21 = 6  # 0110 <-| | |
F3  = 7  # 0111   | | |
V3  = 8  # 1000   | | |  Opposite edges
E03 = 9  # 1001 <-| | |
E13 = 10 # 1010 <---| |
F2  = 11 # 1011       |
E32 = 12 # 1100 <-----|
F1  = 13 # 1101
F0  = 14 # 1110
T   = 15 # 1111

# User-friendly?

E10 = 3
E20 = 5
E12 = 6
E30 = 9
E31 = 10
E23 = 12


# A simplex is oriented like this:  
#     1     
#    /|\    
#   / | \   
#  /  |  \  
# 2---|---3 
#  \  |  /  
#   \ | /   
#    \|/    
#     0
#

def tetrahedron(label=None):
    next = Permutation(cycles=[('01','02','03'),
                               ('12','10','13'),
                               ('20','21','23'),
                               ('30','32','31')])
    opposite = Permutation(cycles=[('01','10'),
                                   ('02','20'),
                                   ('03','30'),
                                   ('12','21'),
                                   ('13','31'),
                                   ('23','32')])
    if label:
        return RibbonGraph([opposite.append_label(label), next.append_label(label)])
    else:
        return RibbonGraph([opposite, next])
    
def tetrahedron_old(label=None):
    next = Permutation(cycles=[((0,1),(0,3),(0,2)),
                               ((1,2),(1,3),(1,0)),
                               ((2,0),(2,3),(2,1)),
                               ((3,0),(3,1),(3,2))])
    opposite = Permutation(cycles=[((0,1),(1,0)),
                                   ((0,2),(2,0)),
                                   ((0,3),(3,0)),
                                   ((1,2),(2,1)),
                                   ((1,3),(3,1)),
                                   ((2,3),(3,2))])
    if label:
        return RibbonGraph([opposite.append_label(label), next.append_label(label)])
    else:
        return RibbonGraph([opposite, next])

def truncate_vertices(ribbon_graph):
    new_opposite = dict(ribbon_graph.opposite)
    next_inverse = ribbon_graph.next.inverse()
    new_next = {}
    for label in ribbon_graph.labels():
        next_label = ribbon_graph.next[label]
        previous_label = next_inverse[label]
        new_next[label]= label+next_label
        new_next[label+previous_label]= label
        new_next[label+next_label]= label+previous_label
        new_opposite[label+next_label]=next_label+label
        new_opposite[next_label+label]=label+ next_label

    new_opposite  = Permutation(new_opposite)
    new_next = Permutation(new_next)
    return RibbonGraph(permutations=[new_opposite,new_next])

    
def truncate_vertices_old(ribbon_graph):
    new_opposite = dict(ribbon_graph.opposite)
    next_inverse = ribbon_graph.next.inverse()
    new_next = {}
    for label in ribbon_graph.labels():
        next_label = ribbon_graph.next[label]
        previous_label = next_inverse[label]
        new_next[label]= (label, next_label)
        new_next[(label,previous_label)]= label
        new_next[(label,next_label)]= (label,previous_label)
        new_opposite[(label,next_label)]=(next_label,label)
        new_opposite[(next_label,label)]=(label, next_label)

    new_opposite  = Permutation(new_opposite)
    new_next = Permutation(new_next)
    return RibbonGraph(permutations=[new_opposite,new_next])


def triangulation_from_pairing(pairing):
    """
    Given pairs of tuples ((directed_edge1,tet_label1),(directed_edge2,tet_label2)), return a Triangulation object with those gluings.
    """
    gluing = Permutation(cycles=pairing)
    tetrahedra = {}
    for pair1, pair2 in pairing:
        directed_edge1, tet_label1 = pair1
        directed_edge2, tet_label2 = pair2

        if tet_label1 not in tetrahedra:
            tetrahedra[tet_label1] = tetrahedron(tet_label1)
        if tet_label2 not in tetrahedra:
            tetrahedra[tet_label2] = tetrahedron(tet_label2)

    U = RibbonGraph([Permutation(), Permutation()])
    for T in tetrahedra.values():
        U = U.union(T)
    return Triangulation(U, gluing)

def triangulation_from_tuples(tuples):
    """
    Given 6-tuples (v1, v2 ,tet_label1, w1, w2 , tet_label2), return a Triangulation object with those gluings.
    """

    tetrahedra = {}
    parsed = []
    for six_tuple in tuples:
        v1, v2 , tet_label1, w1, w2, tet_label2 = six_tuple

        if tet_label1 not in tetrahedra:
            tetrahedra[tet_label1] = tetrahedron(tet_label1)
        if tet_label2 not in tetrahedra:
            tetrahedra[tet_label2] = tetrahedron(tet_label2)

        parsed.append( (((v1,v2),tet_label1), ((w1,w2),tet_label2) ))
    U = RibbonGraph([Permutation(), Permutation()])
    for T in tetrahedra.values():
        U = U.union(T)

    gluing = Permutation(cycles=parsed)
    return Triangulation(U, gluing)





def doubled():
    return [(0,1,0,0,1,1),
            (1,2,0,1,2,1),
            (2,0,0,2,0,1),
            (1,0,0,1,0,1)]


def doubled2():
    return [(0,1,0,1,0,1),
            (1,2,0,2,1,1),
            (2,0,0,0,2,1),
            (1,0,0,0,1,1)]

