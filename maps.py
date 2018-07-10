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
        
class StrandDiagram(object):
    def __init__(self, ribbon_graph, heights):
        self.ribbon_graph = ribbon_graph
        self.heights = heights        
        self._verify_eulerian_and_height_rules()

        
    def _verify_eulerian_and_height_rules(self):
        labels = self.ribbon_graph.labels()
        while labels:
            label = labels.pop()
            vertex = self.ribbon_graph.vertex(label)
            heights_around_vertex = [self.heights[label]]
            for other_vertex_label in vertex[1:]:
                heights_around_vertex.append(self.heights[other_vertex_label])
                labels.remove(other_vertex_label)
            vertex_length = len(heights_around_vertex)
            if (vertex_length % 2) != 0:
                raise Exception("Diagram has vertex of odd degree")
            first_half, second_half = heights_around_vertex[:(vertex_length//2)], heights_around_vertex[(vertex_length//2):]
            if first_half != second_half:
                raise Exception("Strand heights inconsistent around vertex")
            if set(first_half) == set(range(len(first_half))) or set(first_half) == set([0]):
                #checking that first_half is just a permutation of 0,...,len(first_half) (a normal crossing) or all zeros (a virtual crossing)
                continue
            else:
                raise Exception("Strand heights not in allowed pattern.")

    def crossing_type(self, label):
        vertex = self.ribbon_graph.vertex(label)
        vertex_heights = [self.heights[l] for l in vertex]
        if len(set(vertex_heights)) == 1:
            return 'v'
        elif len(set(vertex_heights)) == 2:
            return 'c'
        else:
            return 'm'

        
"""
class StrandDiagram(object):
    def __init__(self, vertices):
        opposite = permutation_from_bijections([v.opposite for v in vertices])
        next = permutation_from_bijections([v.next for v in vertices])
        self.ribbon_graph = RibbonGraph(permutations = [opposite, next])

        self._verify_planarity()

    def _verify_planarity(self):
        pass
"""
    
        
class Link(StrandDiagram):
    def __init__(self, vertices):
        
        super(Link,self).__init__(vertices)

        self._verify_four_valent()
                
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
