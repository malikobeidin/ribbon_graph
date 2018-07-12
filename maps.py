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
    def __init__(self, ribbon_graph, heights, verify=True):
        self.ribbon_graph = ribbon_graph
        self.heights = heights        
        if verify:
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
    
        
class Link(StrandDiagram):
    def __init__(self, vertices=[], PD=[]):
        if PD and not vertices:
            vertices = self._vertices_from_PD(PD)
        opposite = permutation_from_bijections([v.opposite for v in vertices])
        next = permutation_from_bijections([v.next for v in vertices])

        ribbon_graph = RibbonGraph([opposite,next])
        heights = {label: label[1]%2 for label in ribbon_graph.labels()}
        super(Link,self).__init__(ribbon_graph, heights)

        self._verify_valence_and_heights()

    def _vertices_from_PD(self, PD):
        vertices = [MapVertex(i,4) for i in range(len(PD))]
        edge_dict = {}
        for vertex_label, edge_list in enumerate(PD):
            for slot, edge in enumerate(edge_list):
                if edge in edge_dict:
                    old_vertex_label, old_slot = edge_dict[edge]
                    vertices[vertex_label][slot] = vertices[old_vertex_label][old_slot]
                else:
                    edge_dict[edge] = (vertex_label, slot)
        return vertices
                
    def _verify_valence_and_heights(self):
        pass
        
    def spherogram(self):
        vertices = self.ribbon_graph.vertices()
        edges = self.ribbon_graph.edges()
        PD = []
        for v in vertices:
            needs_rotation = False
            if self.heights[v[0]] == 1:
                needs_rotation = True
            vertex_code = []
            for i in v:
                for j, edge in enumerate(edges):
                    if i in edge:
                        vertex_code.append(j)
                        break

            if needs_rotation:
                new_vertex_code = vertex_code[1:]
                new_vertex_code.append(vertex_code[0])
                vertex_code = new_vertex_code
                
            PD.append(vertex_code)
        return spherogram.Link(PD)

def trefoil():
    a, b, c = [MapVertex(x,4) for x in 'abc']
    a[0] = b[3]
    a[1] = b[2]
    b[0] = c[3]
    b[1] = c[2]
    c[0] = a[3]
    c[1] = a[2]
    return Link([a,b,c])
