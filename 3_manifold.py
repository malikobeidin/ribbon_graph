from permutation import Permutation, Bijection
from ribbon_graph_base import *

class Triangulation(object):
    def __init__(self, tetrahedra_ribbon_graph, glued_to, check_consistency = True):
        """
        Specify a triangulation of a 3-manifold with:
               tetrahedra_ribbon_graph: a ribbon graph whose faces are all size 3, and whose connected components are all the tetrahedral map.
               glued_to: a permutation which shows which faces are glued. A directed edge in a tetrahedron uniquely specifies how to glue two faces, by gluing the faces to the left of each directed edge. This permutation must commute with the next_corner permutation of the tetrahedra_ribbon_graph to make it so that faces are glued consistently.
        """
        self.tetrahedra_ribbon_graph = tetrahedra_ribbon_graph
        if check_consistency:
            next_corner = self.tetrahedra_ribbon_graph.next_corner()
            glued_to = next_corner.inverse() * glued_to * next_corner
            commutator = next_corner*glued_to*next_corner.inverse()*glued_to.inverse()
            if not commutator.is_identity():
                raise Exception("Gluing not consistent along faces")
        

    def tetrahedra(self):
        return self.tetrahedra_ribbon_graph.connected_components()

    def vertex_links(self):
        return RibbonGraph([self.glued_to, self.next])

def tetrahedron(tet_label):
    
    next = Permutation(cycles=[((0,1,tet_label),(0,3,tet_label),(0,2,tet_label)),
                               ((1,2,tet_label),(1,3,tet_label),(1,0,tet_label)),
                               ((2,0,tet_label),(2,3,tet_label),(2,1,tet_label)),
                               ((3,0,tet_label),(3,1,tet_label),(3,2,tet_label))])
    opposite = Permutation(cycles=[((0,1,tet_label),(1,0,tet_label)),
                                   ((0,2,tet_label),(2,0,tet_label)),
                                   ((0,3,tet_label),(3,0,tet_label)),
                                   ((1,2,tet_label),(2,1,tet_label)),
                                   ((1,3,tet_label),(3,1,tet_label)),
                                   ((2,3,tet_label),(3,2,tet_label))])
    return RibbonGraph([opposite, next])


def triangulation_from_pairing(pairing):
    """
    Given pairs of tuples ((tet_label1,directed_edge1),(tet_label2,directed_edge2)), return a Triangulation object with those gluings.
    """
    tet_labels = set()
    for pair1, pair2 in pairing:
        tet_label1, directed_edge1 = pair1
        tet_label2, directed_edge2 = pair2
        
    tets = [tetrahedron()]
