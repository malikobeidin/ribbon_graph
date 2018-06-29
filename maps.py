from permutation import Bijection, Permutation, permutation_from_bijections
from ribbon_graph import RibbonGraph
import spherogram

class MapVertex(object):
    def __init__(self, label, num_slots):
        self.label = label
        self.next = Permutation(cycles = [[(self.label,i) for i in range(num_slots)]])
        self.opposite = Bijection()

    def __repr__(self):
        return self.label

    def __getitem__(self, i):
        return (self, i)

    def __setitem__(self, i, other):
        other_vertex, j = other
        if self[i] in self.opposite or other_vertex[j] in other_vertex.opposite:
            raise Exception("Slot already occupied")
        if self[i] == other:
            raise Exception("Can't connect slot to itself")
        self.opposite[(self.label,i)] = (other_vertex.label, j)
        other_vertex.opposite[(other_vertex.label,j)] = (self.label,i)

    def __len__(self):
        return len(self.next)
        
class Map(object):
    def __init__(self, vertices):
        opposite = permutation_from_bijections([v.opposite for v in vertices])
        next = permutation_from_bijections([v.next for v in vertices])
        self.ribbon_graph = RibbonGraph(permutations = [opposite, next])

        self._verify_planarity()

    def _verify_planarity(self):
        pass
        
class EulerianMap(Map):
    def __init__(self, vertices):
        super(EulerianMap,self).__init__(vertices)

    def _verify_eulerian(self):
        for vertex in self.vertices:
            assert len(vertex) % 2 == 0
        
class Link(EulerianMap):
    def __init__(self, vertices):
        
        super(Link,self).__init__(vertices)

    def _verify_four_valent(self):
        for vertex in self.vertices:
            assert len(vertex) == 4
        
    def spherogram(self):
        return spherogram.Link(self.ribbon_graph.PD_code())

def trefoil():
    a, b, c = [MapVertex(x,4) for x in 'abc']
    a[0] = b[3]
    a[1] = b[2]
    b[0] = c[3]
    b[1] = c[2]
    c[0] = a[3]
    c[1] = a[2]
    return Link([a,b,c])
