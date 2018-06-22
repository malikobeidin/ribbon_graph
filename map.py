from permutation import Bijection, Permutation, permutation_from_bijections
from ribbon_graph import RibbonGraph

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

        
class Map(object):
    def __init__(self, vertices):
        opposite = permutation_from_bijections([v.opposite for v in vertices])
        next = permutation_from_bijections([v.next for v in vertices])
        self.ribbon_graph = RibbonGraph(permutations = [opposite, next])

    
