from ribbon_graph_base import RibbonGraph
from permutation import Permutation
from random import shuffle

class MountainRange(object):
    def __init__(self, steps = []):
        s = 0
        for step in steps:
            assert step in [-1,1]
            s += step
            assert s >= 0
        assert s == 0
        self.steps = steps

    def rooted_plane_tree(self):
        """
        Mountain ranges give rooted plane trees in a canonical way.
        """
        
        steps = self.steps
        s = 0
        stack = []
        paired_labels = []
        for i, step in enumerate(steps):
            if step == 1:
                stack.append(i)
            else:
                paired_labels.append( [stack.pop(), i])

        opposite = Permutation(cycles = paired_labels)
        next_corner = Permutation(cycles = [range(len(steps))])
        return RootedPlaneTree(0, permutations = [opposite, next_corner.inverse()*opposite])


def random_mountain_range(n):
    steps = [1]*n
    steps.extend( [-1]*n )
    shuffle(steps)
    s = 0
    for i, step in enumerate(steps):
        s, last_s = s+step, s
        if s < 0  or last_s < 0 :
            steps[i] = -step
    return MountainRange(steps)
    

def remy_random_rooted_binary_plane_tree(n):
    tree = Y()
    for i in range(n):
        tree = tree.insert_leaf_on_edge(tree.random_label(), [(i,j) for j in range(4)])
    return tree

class RootedPlaneTree(RibbonGraph):
    def __init__(self, root, permutations = []):
        super(RootedPlaneTree,self).__init__( permutations = permutations )
        
        assert root in self.labels()

        self.root = root
        
        if len(self.faces())>1:
            raise Exception("Map is not a tree")
        

    def insert_leaf_on_edge(self, label, new_labels):
        picture_to_insert, boundary_labels = open_Y(new_labels)
        pairings = [(label,boundary_labels[0]),(self.opposite[label],boundary_labels[1])]
        new_tree = self.disconnect_edges([label])
        new_tree = new_tree.union(picture_to_insert)
        new_tree = new_tree.connect_edges(pairings)
        return RootedPlaneTree(self.root, [new_tree.opposite, new_tree.next])

    def relabeled(self):
        R = self.relabeled_by_root(self.root)
        return RootedPlaneTree(1,[R.opposite, R.next])

    
def open_Y(new_labels):
    l0,l1,l2,l3 = new_labels
    return RootedPlaneTree(l0, [Permutation(cycles = [(l0,),(l1,),(l2,l3)]) ,
                            Permutation(cycles = [(l0,l1,l2), (l3,)])]), (l0,l1)

def Y():
    return RootedPlaneTree(0, [Permutation(cycles = [(0,3),(1,4),(2,5)]) ,
                            Permutation(cycles = [(0,1,2), (3,), (4,), (5,)])])

