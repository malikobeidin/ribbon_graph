import itertools
from random import choice
from cycle import EmbeddedPath, EmbeddedCycle

class PolygonWithDiagonals(object):
    """
    Numbered clockwise around the boundary, i.e. with the
    exterior face to the left side
    """
    def __init__(self, label, boundary_length, diagonals):
        self.label = label
        self.vertices = range(boundary_length)
        self.diagonals = diagonals
        self.boundary_length = boundary_length
        self._verify_no_diagonal_crossings()
        
    def _verify_no_diagonal_crossings(self):
        for x,y in self.diagonals:
            if x >= self.boundary_length or y >= self.boundary_length or x < 0 or y < 0:
                raise Exception("Diagonal not in correct range.")

        for pair1, pair2 in itertools.combinations(self.diagonals,2):
            x,y = sorted(pair1)
            a,b = sorted(pair2)
            if (a < x < b < y) or (x < a < y < b):
                raise Exception("Diagonals cross.")

    def ribbon_graph(self):
        pass

def polygon_with_diagonals_from_ribbon_graph(ribbon_graph, exterior_label):
    num_vertices = len(ribbon_graph.vertices())
    boundary_cycle = EmbeddedCycle(ribbon_graph, exterior_label, turn_degrees = [-1]*num_vertices)
    diagonals = []
    for l1, l2 in ribbon_graph.edges():
        if (l1 in boundary_cycle.labels) or (l2 in boundary_cycle.labels):
            continue
        l1_vertex = None
        l2_vertex = None
        for i, label in enumerate(boundary_cycle.labels[:-1]):
            if l1 in ribbon_graph.vertex(label):
                l1_vertex = i
            if l2 in ribbon_graph.vertex(label):
                l2_vertex = i
            if l1_vertex and l2_vertex:
                break
        diagonals.append((l1_vertex,l2_vertex))
    polygon_label = str(boundary_cycle.labels[:-1])
    return PolygonWithDiagonals(polygon_label, num_vertices, diagonals)
            
    


class CycleTree(object):
    def __init__(self, cycle, max_length):
        self.left = None
        self.right = None
        self.cycle = cycle
        self.max_length = max_length
        self.num_nonboundary_vertices = len(cycle.left_side().vertices())
        if self.is_splittable():
            self.split()

    def __repr__(self):
        return "T({})".format(self.cycle)
            
    def is_splittable(self):
        return len(self.cycle) < self.num_nonboundary_vertices

    def split(self):
        max_length = min(self.max_length, self.num_nonboundary_vertices+1)
        cycle = self.cycle
        possible_splitting_path_starts = interior_pointing_paths(cycle)
        splitting_path = None
        for path in possible_splitting_path_starts:
            splitting_path = find_splitting_path(cycle, path, max_length)
            if splitting_path:
                break
        cycle1, cycle2 = cycle.split_along_path(splitting_path)
        self.left = CycleTree(cycle1, max_length)
        self.right = CycleTree(cycle2, max_length)
            
    
    def split_old(self):
        cycle = self.cycle
        ribbon_graph = cycle.ribbon_graph
        vertices = ribbon_graph.vertices()
        boundary_labels = set(cycle.labels)
        num_vertices = len(vertices)
        interior_labels = ribbon_graph.labels() - boundary_labels
        
        num_boundary_vertices = len(cycle)
        start_label = cycle.start_label
        subcycles = ribbon_graph.search_for_embedded_cycle_with_start_and_goal(start_label, interior_labels, num_vertices)

        subcycle = max(subcycles, key = len)
        leftover_subcycle = cycle.oriented_sum(subcycle.reversed())

        subgraph = subcycle.left_side()
        leftover_subgraph = leftover_subcycle.left_side()

        subcycle_pushed_to_subgraph = EmbeddedCycle(subgraph, subcycle.start_label, labels = subcycle.labels)
        leftover_subcycle_pushed_to_subgraph = EmbeddedCycle(leftover_subgraph, leftover_subcycle.start_label, labels = leftover_subcycle.labels)
        
        self.left = CycleTree(subcycle_pushed_to_subgraph)
        self.right = CycleTree(leftover_subcycle_pushed_to_subgraph)

    def leaves(self):
        leaves = []
        if self.left:
            leaves.extend(self.left.leaves())
            leaves.extend(self.right.leaves())
            return leaves
        else:
            return [self]


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


def interior_pointing_paths(cycle):
    boundary_vertices = set([frozenset(cycle.ribbon_graph.vertex(l)) for l in cycle.labels])
    seed_paths = []
    for label_list in cycle.left_side_labels():
        for label in label_list:
            op_label = cycle.ribbon_graph.opposite[label]
            is_boundary = False
            for boundary_vertex in boundary_vertices:
                if op_label in boundary_vertex:
                    is_boundary = True
                    break
            if not is_boundary:
                seed_paths.append(EmbeddedPath(cycle.ribbon_graph, label, labels = [label]))
    return seed_paths

def find_splitting_path(cycle, seed_path, max_length):
    """
    Find paths starting on the cycle, going through the left side of the cycle, 
    and back to the cycle again.
    """
    seed_paths = [seed_path]
    boundary_vertices = set([frozenset(cycle.ribbon_graph.vertex(l)) for l in cycle.labels])            
    longest_splitting_path = None
    biggest_length = 0
    for i in range(max_length-1):
        new_paths = []
        for path in seed_paths:
            new_paths.extend(path.one_step_continuations())

        if new_paths:
            for path in new_paths:
                next_vertex = frozenset(path.next_vertex())
                if (next_vertex in boundary_vertices) and (not path.is_completable_to_cycle()):
                    if len(path)>biggest_length:
                        biggest_length = len(path)
                        longest_splitting_path = path
                else:
                    seed_paths.append(path)

        else:
            #no paths found
            break
    return longest_splitting_path

        
