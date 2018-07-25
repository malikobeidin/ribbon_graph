import itertools

class PolygonWithDiagonals(object):
    def __init__(self, label, boundary_length, diagonals):
        self.label
        self.vertices = range(boundary_length)
        self.diagonals = diagonals

    def _verify_no_diagonal_crossings(self):
        for x,y in diagonals:
            if (not 0 <= x < boundary_length) or (not 0 <= y < boundary_length):
                raise Exception("Diagonal not in correct range.")

        for pair1, pair2 in itertools.combinations(diagonals,2):
            x,y = sorted(pair1)
            a,b = sorted(pair2)
            if (a < x and b < y) or (x < a and y < b):
                raise Exception("Diagonals cross.")

            

class PolygonalDecomposition(object):
    def __init__(self, ribbon_graph, start_label, max_length):
        cycles = self.search_for_long_embedded_cycles_through(start_point, max_length)
        
        
        
