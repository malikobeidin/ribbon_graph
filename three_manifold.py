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

def tetrahedron(label=None):
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

